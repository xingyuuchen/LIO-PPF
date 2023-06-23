#pragma once

#include <ctime>
#include <set>
#include <list>
#include <vector>
#include <glog/logging.h>
#include "sac_model.h"
//#include "timer.h"


template<class SacModelT>
class RansacWithPca {
  public:
    using SampleConsensusModelPtr = typename SacModelT::Ptr;
    using Ptr = std::shared_ptr<RansacWithPca<SacModelT>>;
    using ConstPtr = std::shared_ptr<const RansacWithPca<SacModelT>>;
    using ModelType = typename SacModelT::ModelType;
    
    
    explicit RansacWithPca(const SampleConsensusModelPtr &model, int n_max_models,
                           int n_least_inliers, bool random = false);
    
    bool ComputeModel(std::vector<ModelType> &models_init_guess);
    bool ComputeModelsFromInitGuess(std::vector<ModelType> &models_init_guess);
    bool ComputeOneModel();
    
    inline std::vector<ModelType> &AllModelsCoef() { return all_models_coef_; }
    
    inline int NumModels() { return all_models_coef_.size(); }
    
    inline void SetDistanceThreshold(float threshold) { threshold_ = threshold; }
    inline double GetDistanceThreshold() { return threshold_; }
    
    inline void setMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
    inline int getMaxIterations() { return max_iterations_; }
    
    // probability: the desired probability of choosing at least one
    // sample free from outliers, 99% by default.
    inline void SetProbability(double probability) { probability_ = probability; }
    inline double GetProbability() { return probability_; }
    
    /** \brief Refine the model found.
      * This loops over the model coefficients and optimizes them together
      * with the set of inliers, until the change in the set of inliers is minimal.
      * \param[in] sigma standard deviation multiplier for considering a sample as inlier (Mahalanobis distance)
      * \param[in] max_iterations the maxim number of iterations to try to refine in case the inliers keep on changing
      */
    bool RefineModel(ModelType &model_coef, int max_iterations, 
                     int *n_inliers_after_refine = nullptr, double sigma = 3.0);
    
    inline void GetRandomSamples(const std::shared_ptr<std::vector<int>> &indices,
                                 size_t nr_samples,
                                 std::set<int> &indices_subset) {
        indices_subset.clear();
        while (indices_subset.size() < nr_samples) {
            indices_subset.insert((*indices)[static_cast<int> (static_cast<double>(indices->size()) * Rnd())]);
        }
    }
    
    inline int GetIterations() { return iterations_; }
    
    inline int InitTotalPoints() { return (int) init_total_samples_; }
    
    inline int CurrTotalPoints() { return (int) sac_model_->NumPoints(); }
    
    inline void SetSampleConsensusModel(const SampleConsensusModelPtr &model) {
        sac_model_ = model;
    }
    
    inline SampleConsensusModelPtr GetSampleConsensusModel() const { return sac_model_; }
  
  private:
    inline double Rnd() { return rng_->operator()(); }
    
    size_t n_max_models_;
    int n_least_inliers_;
    std::vector<ModelType> all_models_coef_;
    std::vector<int> all_models_n_inliers_;
    size_t init_total_samples_;
    
    SampleConsensusModelPtr sac_model_;
    std::vector<int> model_;
    ModelType model_coef_;
    double probability_;
    int iterations_;
    float threshold_;
    int max_iterations_;
    // random number generator algorithm.
    boost::mt19937 rng_alg_;
    // random number generator distribution.
    std::shared_ptr<boost::uniform_01<boost::mt19937>> rng_;
};


template<typename SacModelT>
RansacWithPca<SacModelT>::RansacWithPca(const RansacWithPca::SampleConsensusModelPtr &model,
                int n_max_models, int n_least_inliers, bool random /*= false */)
        : n_max_models_(n_max_models), n_least_inliers_(n_least_inliers)
        , init_total_samples_(model->NumPoints()), sac_model_(model), model_()
        , model_coef_(), probability_(0.99), iterations_(0)
        , threshold_(std::numeric_limits<float>::max()), max_iterations_(0)
        , rng_alg_(), rng_(new boost::uniform_01<boost::mt19937>(rng_alg_)) {
    
    // Create a random number generator object
    rng_->base().seed(random ? static_cast<unsigned>(std::time(nullptr)) : 12345u);
}


template<typename SacModelT>
bool
RansacWithPca<SacModelT>::ComputeModel(std::vector<ModelType> &models_init_guess) {
    DLOG_ASSERT(threshold_ != std::numeric_limits<float>::max());
    all_models_coef_.clear();
    all_models_n_inliers_.clear();

    if (!models_init_guess.empty()) {
        ComputeModelsFromInitGuess(models_init_guess);
    }
    while (true) {
        if (sac_model_->NumPoints() < init_total_samples_ / 4 ||
                all_models_coef_.size() >= n_max_models_) {
            break;
        }
        bool succeed = ComputeOneModel();
        if (!succeed) {
            // Iteration will end once the Ransac fails, because Ransac finds the
            // best model for now, and the best model is already not good,
            // let alone subsequent ones.
            break;
        }
    }

    std::vector<int> arg_min(all_models_n_inliers_.size());
    std::iota(arg_min.begin(), arg_min.end(), 0);
    DLOG_ASSERT(all_models_n_inliers_.size() == all_models_coef_.size());
    std::sort(arg_min.begin(), arg_min.end(), [&] (int &lrs, int &rhs) {
        /* descending */
        return all_models_n_inliers_[lrs] > all_models_n_inliers_[rhs];
    });
    decltype(all_models_coef_) sort_models;
    for (auto &idx : arg_min) {
        sort_models.emplace_back(all_models_coef_[idx]);
    }
    all_models_coef_ = sort_models;
    return true;
}

template<typename SacModelT>
bool
RansacWithPca<SacModelT>::ComputeModelsFromInitGuess(std::vector<ModelType> &models_init_guess) {
    const int n_init_guess = models_init_guess.size();
    int n_accept = 0;
    int n_remove_pts = 0;
    
    for (auto &model_guess : models_init_guess) {
        // refine from init guess
        int n_inliers = sac_model_->CountWithinDistance(model_guess, threshold_);
        if (n_inliers < n_least_inliers_) {
            continue;
        }
        int n_inliers_after_refine;
        if (!RefineModel(model_guess, 5, &n_inliers_after_refine)) {
            continue;
        }
        ++n_accept;
        all_models_coef_.template emplace_back(model_guess);
        all_models_n_inliers_.emplace_back(n_inliers_after_refine);

        int n_remove = sac_model_->RemoveWithinDistance(model_guess, threshold_, nullptr, true);
        n_remove_pts += n_remove;
    }
    return true;
}

template<typename SacModelT>
bool
RansacWithPca<SacModelT>::ComputeOneModel() {
    iterations_ = 0;
    int n_best_inliers = -1;
    double k = 1.0;
    
    std::vector<int> samples;
    ModelType model_coef;
    
    double log_probability = log(1.0 - probability_);
    double one_over_indices = 1.0 / (double) sac_model_->getIndices()->size();
    
    unsigned skip_cnt = 0;
    const unsigned max_skip = max_iterations_ * 10;
    
    while (iterations_ < k && skip_cnt < max_skip) {
        sac_model_->GetSamples(iterations_, samples);
    
        if (samples.empty() || !sac_model_->ComputeModelCoef(samples, model_coef)) {
            ++skip_cnt;
            continue;
        }
        int n_inliers = sac_model_->CountWithinDistance(model_coef, threshold_);
        if (n_inliers > n_best_inliers) {
            n_best_inliers = n_inliers;
            model_ = samples;
            model_coef_ = model_coef;
            double w = static_cast<double>(n_best_inliers) * one_over_indices;
            double p_no_outliers = 1.0 - pow(w, static_cast<double>(samples.size()));
            p_no_outliers = std::max(std::numeric_limits<double>::epsilon(),
                                       p_no_outliers);       // Avoid division by -Inf
            p_no_outliers = std::min(1.0 - std::numeric_limits<double>::epsilon(),
                                       p_no_outliers);   // Avoid division by 0.
            k = log_probability / log(p_no_outliers);
        }
        
        if (++iterations_ > max_iterations_) {
            break;
        }
    }
    if (n_best_inliers < n_least_inliers_ || model_.empty()) {
        return false;
    }
    
    RefineModel(model_coef_, 3, &n_best_inliers);
    all_models_coef_.emplace_back(model_coef_);
    all_models_n_inliers_.emplace_back(n_best_inliers);

    int n_rm = sac_model_->RemoveWithinDistance(model_coef_, threshold_, nullptr, true);
    return true;
}


template<class SacModelT>
bool RansacWithPca<SacModelT>::RefineModel(ModelType &model_coef,
                                           const int max_iterations,
                                           int *n_inliers_after_refine /* = nullptr*/,
                                           const double sigma/* = 3.0*/) {
    double thresh_sqr = threshold_ * threshold_, error_thresh = threshold_;
    const double sigma_sqr = sigma * sigma;
    int refine_iter = 0;
    bool inlier_changed = false, oscillating = false;
    std::vector<size_t> inliers_sizes;
    std::vector<int> new_inliers, prev_inliers;
    ModelType new_model_coef;
    sac_model_->SelectWithinDistance(model_coef, threshold_, prev_inliers);
    do {
        sac_model_->OptimizeModel(prev_inliers, new_model_coef);
        inliers_sizes.push_back(prev_inliers.size());
        
        // Select the new inliers based on the optimized coefficients and new threshold
        sac_model_->SelectWithinDistance(new_model_coef, error_thresh, new_inliers);
        
        if (new_inliers.empty()) {
            if (++refine_iter >= max_iterations)
                break;
            continue;
        }
        
        // Estimate the variance and the new threshold
        double variance = sac_model_->ComputeVariance();
        error_thresh = sqrt(std::min(thresh_sqr, sigma_sqr * variance));
        
        inlier_changed = prev_inliers.size() != new_inliers.size();
        if (inlier_changed) {
            // check if the number of inliers is oscillating in between two values
            if (inliers_sizes.size() >= 4) {
                size_t n_size = inliers_sizes.size();
                if (inliers_sizes[n_size - 1] == inliers_sizes[n_size - 3] &&
                    inliers_sizes[n_size - 2] == inliers_sizes[n_size - 4]) {
                    oscillating = true;
                    break;
                }
            }
        }
        prev_inliers = new_inliers;
        
    } while (inlier_changed && ++refine_iter < max_iterations);
    
    DLOG_ASSERT(!new_inliers.empty()) << "Refining failed: got an empty set of inliers!";
    if (oscillating) {
        DLOG(INFO) << "Detected oscillations in the model refinement.";
        return false;
    }
    
    model_coef = new_model_coef;
    if (n_inliers_after_refine) {
        *n_inliers_after_refine = new_inliers.size();
    }
    return true;
}



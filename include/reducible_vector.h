#pragma once

#include <vector>
#include <omp.h>
#include <glog/logging.h>



/**
 * Reducible std::vector with fast push_back.
 */
template <class T>
class OmpReducibleVector {
  public:
    
    OmpReducibleVector(int capability_each_thread, std::vector<T> *vec);
    
    ~OmpReducibleVector();
    
    inline void FastPushBack(T &&element) {
#ifdef _OPENMP
//        DLOG_ASSERT(!is_master_vec_) << "Do not invoke FastPushBack on the master, "
//                                        "as the master vector is only for reduction.";
        // To ensure efficiency, we do not check capability,
        // it is the user's responsibility to ensure that no
        // array index out of bounds exception occurs.
        vec_->operator[](n_elements_++) = element;
#else
        // if omp is disabled, act like normal std::vector
        vec_->template emplace_back(element);
#endif
    }
    
    inline void FastPushBack(T &element) {
#ifdef _OPENMP
//        DLOG_ASSERT(!is_master_vec_) << "Do not invoke FastPushBack on the master, "
//                                        "as the master vector is only for reduction.";
        // To ensure efficiency, we do not check capability,
        // it is the user's responsibility to ensure that no
        // array index out of bounds exception occurs.
        vec_->operator[](n_elements_++) = element;
#else
        // if omp is disabled, act like normal std::vector
        vec_->template emplace_back(element);
#endif
    }
    
    inline int CapabilityForNonMaster() { return capability_non_master_; }
    
    void ReductionMergeWith(OmpReducibleVector<T> &another);


  private:
    std::vector<T>    * vec_;
    int                 n_elements_;
    int                 capability_non_master_;
    bool                is_master_vec_;
};



template <class T>
OmpReducibleVector<T>::OmpReducibleVector(int capability_each_thread,
                                          std::vector<T> *vec)
        : vec_(vec)
        , n_elements_(0)
        , capability_non_master_(capability_each_thread)
        , is_master_vec_(vec != nullptr) {
#ifndef _OPENMP
    LOG_ASSERT(is_master_vec_) << "vec is NULL!";
#endif
    if (!is_master_vec_) {  // i.e. vec == nullptr
        // If this is not the master vector in Omp,
        // create a temp vector for the thread to hold elements.
        vec_ = new std::vector<T>(capability_each_thread);
    }
}


template<class T>
void OmpReducibleVector<T>::ReductionMergeWith(OmpReducibleVector<T> &another) {
#ifndef _OPENMP
    LOG_ASSERT(false) << "ReductionMergeWith can only be invoked by Omp!";
#endif
    DLOG_ASSERT(is_master_vec_ && !another.is_master_vec_)
        << "The master can only reduce with a non-master";
    LOG_ASSERT(another.n_elements_ <= another.capability_non_master_) <<
        "Array index out of bounds: (" << another.n_elements_ << " / " <<
        another.capability_non_master_ << ")!!";
    
    another.vec_->resize(another.n_elements_);
    
    vec_->insert(vec_->end(), another.vec_->begin(), another.vec_->end());
    n_elements_ += another.n_elements_;
}


template<class T>
OmpReducibleVector<T>::~OmpReducibleVector() {
    if (!is_master_vec_) {
        delete vec_, vec_ = nullptr;
    }
}



#pragma omp declare reduction(merge_std_vec_i: std::vector<int>: \
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = decltype(omp_orig)())

#pragma omp declare reduction(merge_std_vec_d: std::vector<double>: \
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) \
    initializer(omp_priv = decltype(omp_orig)())
    
    
#pragma omp declare reduction(merge_reducible_vec_i: OmpReducibleVector<int>: \
    omp_out.ReductionMergeWith(omp_in)) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.CapabilityForNonMaster(), nullptr))
    
#pragma omp declare reduction(merge_reducible_vec_f: OmpReducibleVector<float>: \
    omp_out.ReductionMergeWith(omp_in)) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.CapabilityForNonMaster(), nullptr))
    
#pragma omp declare reduction(merge_reducible_vec_d: OmpReducibleVector<double>: \
    omp_out.ReductionMergeWith(omp_in)) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.CapabilityForNonMaster(), nullptr))


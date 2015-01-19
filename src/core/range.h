#pragma once

#include <assert.h>
#include <initializer_list>

namespace nnFit {

template<size_t N>
struct RangeND {
    RangeND() {
        for (size_t i = 0; i < N; ++i)
            sizes[i] = 0;
    }
    
    RangeND(std::initializer_list<size_t> init) {
        assert(N == init.size());
        for (size_t i = 0; i < N; ++i)
            sizes[i] = init.begin()[i];
    }
    
    RangeND<N> &operator = (const RangeND<N> &other) {
        for (size_t i = 0; i < N; ++i)
            sizes[i] = other.sizes[i];
        return *this;
    }
    
    inline size_t *data() {
        return sizes;
    }
    
    inline const size_t *data() const {
        return sizes;
    }
    
    inline size_t operator [](size_t i) const {
        assert(i < N);
        return sizes[i];
    }
protected:
    size_t sizes[N];
};
    
struct Range2D : RangeND<2> {
    Range2D() { }
    Range2D(size_t x, size_t y) {
        sizes[0] = x;
        sizes[1] = y;
    }
};
    
} // namespace nnFit

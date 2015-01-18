#pragma once

#include <stdint.h>

namespace nnFit {

struct ValueType {
    enum struct ElementType {
        Float,
        Uint8,
        Uint16,
        Uint32
    };
    
    // Typesafe enumarations accessible from struct scope.
    static const ElementType Float = ElementType::Float;
    static const ElementType Uint8 = ElementType::Uint8;
    static const ElementType Uint16 = ElementType::Uint16;
    static const ElementType Uint32 = ElementType::Uint32;
    
    constexpr ValueType(ElementType type) : etype(type) {
    }
    
    inline ElementType type() const {
        return etype;
    }
    
    bool operator == (const ValueType &other) const {
        return etype == other.etype;
    }
    
    bool operator == (const ElementType other) const {
        return etype == other;
    }
    
    bool operator != (const ValueType &other) const {
        return etype != other.etype;
    }
    
    size_t size() const {
        switch (etype) {
            case ElementType::Float: return sizeof(float);
            case ElementType::Uint8: return sizeof(uint8_t);
            case ElementType::Uint16: return sizeof(uint16_t);
            case ElementType::Uint32: return sizeof(uint32_t);
        }
    }
private:
    ElementType etype;
};

namespace util {

    template<typename T>
    struct ValueTypeSelector {
    };

    template<>
    struct ValueTypeSelector<float> {
        static const auto value = ValueType::Float;
    };

    template<>
    struct ValueTypeSelector<uint8_t> {
        static const auto value = ValueType::Uint8;
    };

    template<>
    struct ValueTypeSelector<uint16_t> {
        static const auto value = ValueType::Uint16;
    };

    template<>
    struct ValueTypeSelector<uint32_t> {
        static const auto value = ValueType::Uint32;
    };
}

template<typename T>
constexpr ValueType valueType() {
    return util::ValueTypeSelector<T>::value;
}

} // namespace nnFit

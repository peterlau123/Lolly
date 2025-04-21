#pragma once

#include "../../Utility.h"

namespace Lolly{

    namespace parallel{
    class ReduceType{
    public:
        enum Type{
            ADD,
            MULTIPLY,
            MAX,
            MIN,
            AND,
            OR,
            XOR
        };
        static const char* toString(Type type){
            switch(type){
                case ADD: return "ADD";
                case MULTIPLY: return "MULTIPLY";
                case MAX: return "MAX";
                case MIN: return "MIN";
                case AND: return "AND";
                case OR: return "OR";
                case XOR: return "XOR";
                default: return "UNKNOWN";
            }
        }
    };
    /*
    * This function performs a reduction operation on the input array.
    * It takes an array of floats, an output pointer, the size of the array,
    * and the type of reduction to perform.
    * @param[in]:input
    * @param[in]:size
    * @param[out]:out, need to allocate manually outside this function
    */
    static void reduce(_IN_ float* input, _INOUT_ float** out, _IN_ size, _IN_ ReduceType::Type type);
}

}
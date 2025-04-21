#include "lolly/parallel/basic_arithmetic.h"






using namespace Lolly::parallel;

void Lolly::reduce(float* left, float** out, int size, ReduceType::Type type) {
        if(nullptr==left||out==nullptr||size<=0){
            return;
        }

}
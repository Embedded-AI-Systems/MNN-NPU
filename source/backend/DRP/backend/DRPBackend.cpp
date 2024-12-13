#include "TRTBackend.hpp"

#include <stdlib.h>
#include <map>
#include <string.h>

namespace MNN{
    Execution* DRPBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op){
        auto map = getCreatorMap();
        auto iter =map->find(op->type());
        
        if(iter == map->end()){
            MNN_ERROR("map not find !!! \n");
            if(op->type()!=nullptr){
                if(op->name()!=nullptr){
                    MNN_PRINT("[NPU] Don't support type %d, %s",op->type(),op->name());
                }
            }
            return nullptr;
        }

        auto exe = iter->second->onCreate(input,output,op,this);
        if (nullptr == exe) {
            MNN_ERROR("nullptr == exe !!! \n");
            if(op!=nullptr){
                if(op->name()!=nullptr){
                    MNN_PRINT("[NPU] The creator don't support type %d, %s\n",op->type(),op->name()->c_str());
                }
            }
            return nullptr;
        }
        return exe;
    }
    void DRPBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const{
        bool isConst = (TensorUtils::getDescribe(srcTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT ||TensorUtils::getDescribe(dstTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        
    }
    bool DRPBackend::onClearBuffer(){
        return true;
    }
}
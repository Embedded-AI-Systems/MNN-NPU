#ifndef MNN_DRPBackend_H
#define MNN_DRPBackend_H

// libs to add

#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>

#include "MNN_generated.h"

#include <stdio.h>
#include <map>
#include <memory>

using namespace std;
namespace MNN{


class DRPBackend : public Backend {
    public:
        
        DRPBackend(const NPURuntime* runtime);
        virtual ~DRPBackend();
        
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
        
        virtual void onExecuteBegin() const override;
        
        virtual void onExecuteEnd() const override;
        
        virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
        
        virtual bool onClearBuffer() override;
        
        virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
        
        virtual void onResizeBegin() override;

        virtual ErrorCode onResizeEnd() override;
}   

}
#endif
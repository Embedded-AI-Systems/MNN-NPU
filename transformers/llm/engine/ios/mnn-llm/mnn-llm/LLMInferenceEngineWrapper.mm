//
//  LLMInferenceEngineWrapper.m
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//
#include <functional>
#import "LLMInferenceEngineWrapper.h"
#include <MNN/llm/llm.hpp>
using namespace MNN::Transformer;

const char* GetMainBundleDirectory() {
    NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
    return [bundleDirectory UTF8String];
}

@implementation LLMInferenceEngineWrapper {
    std::shared_ptr<Llm> llm;
}

- (instancetype)initWithCompletionHandler:(ModelLoadingCompletionHandler)completionHandler {
    self = [super init];
    if (self) {
        // 在后台线程异步加载模型
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            BOOL success = [self loadModel]; // 假设loadModel方法加载模型并返回加载的成功或失败
            // 切回主线程回调
            dispatch_async(dispatch_get_main_queue(), ^{
                completionHandler(success);
            });
        });
    }
    return self;
}

- (BOOL)loadModel {
    if (!llm) {
        std::string model_dir = GetMainBundleDirectory();
        std::string config_path = model_dir + "/config.json";
        llm.reset(Llm::createLLM(config_path));
        NSString *tempDirectory = NSTemporaryDirectory();
        llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\", \"use_mmap\":true}");
        llm->load();
    }
    return YES;
}
// Llm start
// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }
private:
    CallBack callback_ = nullptr;
};
- (void)processInput:(NSString *)input withStreamHandler:(StreamOutputHandler)handler {
    LlmStreamBuffer::CallBack callback = [handler](const char* str, size_t len) {
        if (handler) {
            NSString *nsOutput = [NSString stringWithUTF8String:str];
            handler(nsOutput);
        }
    };
    LlmStreamBuffer streambuf(callback);
    std::ostream os(&streambuf);
    if (std::string([input UTF8String]) == "benchmark") {
        // do benchmark
        std::string model_dir = GetMainBundleDirectory();
        std::string prompt_file = model_dir + "/bench.txt";
        std::ifstream prompt_fs(prompt_file);
        std::vector<std::string> prompts;
        std::string prompt;
        while (std::getline(prompt_fs, prompt)) {
            // prompt start with '#' will be ignored
            if (prompt.substr(0, 1) == "#") {
                continue;
            }
            std::string::size_type pos = 0;
            while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
                prompt.replace(pos, 2, "\n");
                pos += 1;
            }
            prompts.push_back(prompt);
        }
        for (int i = 0; i < prompts.size(); i++) {
            llm->response(prompts[i], &os, "\n");
        }
        os << "\n#################################\n"
           << "prompt tokens num  = " << llm->getTotalPromptLen() << "\n"
           << "decode tokens num  = " << llm->getTotalDecodeLen() << "\n"
           << "prefill time = " << std::fixed << std::setprecision(2) << llm->getTotalPrefillTime() << " s\n"
           << " decode time = " << std::fixed << std::setprecision(2) << llm->getTotalDecodeTime() << " s\n"
           << "prefill speed = " << std::fixed << std::setprecision(2) << llm->average_prefill_speed() << " tok/s\n"
           << " decode speed = " << std::fixed << std::setprecision(2) << llm->average_decode_speed() << " tok/s\n"
           << "##################################\n";
        os << "<eop>";
    } else {
        llm->response([input UTF8String], &os, "<eop>");
    }
}

- (void)dealloc {
    llm.reset();
}
@end
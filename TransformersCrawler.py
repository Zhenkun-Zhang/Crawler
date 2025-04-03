from bs4 import BeautifulSoup
import re, os, json, requests
from Tools import *

mainClasses = ['agent', 'auto', 'backbones', 'callback', 'configuration', 
               'data_collator', 'keras_callbacks', 'logging', 'model', 
               'text_generation', 'onnx', 'optimizer_schedules', 'output', 
               'pipelines', 'peft', 'executorch', 
               'processors', 'quantization', 'tokenizer', 'trainer', 'deepspeed', 
               'feature_extractor', 'image_processor']

models = ["albert", "bamba", "bart", "barthez", "bartpho", "bert", "bert-generation", 
          "bert-japanese", "bertweet", "big_bird", "bigbird_pegasus", "biogpt", "blenderbot", 
          "blenderbot-small", "bloom", "bort", "byt5", "camembert", "canine", "codegen", "code_llama", 
          "cohere", "cohere2", "convbert", "cpm", "cpmant", "ctrl", "dbrx", "deberta", "deberta-v2", "deepseek_v3", 
          "dialogpt", "diffllama", "distilbert", "dpr", "electra", "encoder-decoder", "ernie", "ernie_m", "esm", "falcon", 
          "falcon3", "falcon_mamba", "flan-t5", "flan-ul2", "flaubert", "fnet", "fsmt", "funnel", "fuyu", "gemma", 
          "gemma2", "glm", "openai-gpt", "gpt_neo", "gpt_neox", "gpt_neox_japanese", "gptj", "gpt2", "gpt_bigcode", 
          "gptsan-japanese", "gpt-sw3", "granite", "granitemoe", "granitemoeshared", "granitevision", "helium", "herbert", 
          "ibert", "jamba", "jetmoe", "jukebox", "led", "llama", "llama2", "llama3", "longformer", "longt5", "luke", "m2m_100", 
          "madlad-400", "mamba", "mamba2", "marian", "markuplm", "mbart", "mega", "megatron-bert", "megatron_gpt2", "mistral", 
          "mistral3", "mixtral", "mluke", "mobilebert", "modernbert", "mpnet", "mpt", "mra", "mt5", "mvp", "myt5", "nemotron", 
          "nezha", "nllb", "nllb-moe", "nystromformer", "olmo", "olmo2", "olmoe", "open-llama", "opt", "pegasus", "pegasus_x", 
          "persimmon", "phi", "phi3", "phi4_multimodal", "phimoe", "phobert", "plbart", "prophetnet", "qdqbert", "qwen2", "qwen2_moe", 
          "qwen3", "qwen3_moe", "rag", "realm", "recurrent_gemma", "reformer", "rembert", "retribert", "roberta", "roberta-prelayernorm", 
          "roc_bert", "roformer", "rwkv", "splinter", "squeezebert", "stablelm", "starcoder2", "switch_transformers", "t5", "t5v1.1", "tapex", 
          "transfo-xl", "ul2", "umt5", "xmod", "xglm", "xlm", "xlm-prophetnet", "xlm-roberta", "xlm-roberta-xl", "xlm-v", "xlnet", "yoso", "zamba", 
          "zamba2", "beit", "bit", "conditional_detr", "convnext", "convnextv2", "cvt", "dab-detr", "deformable_detr", "deit", "depth_anything", 
          "depth_anything_v2", "depth_pro", "deta", "detr", "dinat", "dinov2", "dinov2_with_registers", "dit", "dpt", "efficientformer", 
          "efficientnet", "focalnet", "glpn", "hiera", "ijepa", "imagegpt", "levit", "mask2former", "maskformer", "mobilenet_v1", "mobilenet_v2",
          "mobilevit", "mobilevitv2", "nat", "poolformer", "prompt_depth_anything", "pvt", "pvt_v2", "regnet", "resnet", "rt_detr", "rt_detr_v2", 
          "segformer", "seggpt", "superglue", "superpoint", "swiftformer", "swin", "swinv2", "swin2sr", "table-transformer", "textnet", "timm_wrapper", 
          "upernet", "van", "vit", "vit_hybrid", "vitdet", "vit_mae", "vitmatte", "vit_msn", "vitpose", "yolos", "zoedepth", 
          "audio-spectrogram-transformer", "bark", "clap", "dac", "encodec", "fastspeech2_conformer", "hubert", "mctct", "mimi", 
          "mms", "moonshine", "moshi", "musicgen", "musicgen_melody", "pop2piano", "seamless_m4t", "seamless_m4t_v2", "sew", 
          "sew-d", "speech_to_text", "speech_to_text_2", "speecht5", "unispeech", "unispeech-sat", "univnet", "vits", "wav2vec2", 
          "wav2vec2-bert", "wav2vec2-conformer", "wav2vec2_phoneme", "wavlm", "whisper", "xls_r", "xlsr_wav2vec2", "timesformer", 
          "videomae", "vivit", "align", "altclip", "aria", "aya_vision", "blip", "blip-2", "bridgetower", "bros", "chameleon", 
          "chinese_clip", "clip", "clipseg", "clvp", "colpali", "data2vec", "deplot", "donut", "emu3", "flava", "gemma3", "git", 
          "got_ocr2", "grounding-dino", "groupvit", "idefics", "idefics2", "idefics3", "instructblip", "instructblipvideo", 
          "kosmos-2", "layoutlm", "layoutlmv2", "layoutlmv3", "layoutxlm", "lilt", "llava", "llava_next", "llava_next_video", 
          "llava_onevision", "lxmert", "matcha", "mgp-str", "mllama", "nougat", "omdet-turbo", "oneformer", "owlvit", "owlv2", 
          "paligemma", "perceiver", "pix2struct", "pixtral", "qwen2_5_vl", "qwen2_audio", "qwen2_vl", "sam", "shieldgemma2", 
          "siglip", "siglip2", "smolvlm", "speech-encoder-decoder", "tapas", "trocr", "tvlt", "tvp", "udop", "video_llava", 
          "vilt", "vipllava", "vision-encoder-decoder", "vision-text-dual-encoder", "visual_bert", "xclip", "decision_transformer", 
          "trajectory_transformer", "autoformer", "informer", "patchtsmixer", "patchtst", "time_series_transformer", "graphormer"]
print(len(models))
internalHelpers = ['modeling_utils', 'pipelines_utils', 'tokenization_utils', 
                   'trainer_utils', 'generation_utils', 'image_processing_utils', 
                   'audio_utils', 'file_utils', 'time_series_utils',
                   'model_debugging_utils', ]

def solve(root, save_path, version):
    apinum = 0
    
    for href in models:
        url = root + "model_doc/" + href
        apinum = getInformation(url, apinum, save_path + "models/", version, href)
    
    # for Class in mainClasses:
    #     url = root + "main_classes/" + Class
    #     apinum = getInformation(url, apinum, save_path + "main_classes/", version, Class)
    #     url = root + "model_doc/" + Class
    #     apinum = getInformation(url, apinum, save_path + "main_classes/", version, Class)

    # for internal in internalHelpers:
    #     url = root + "internal/" + internal
    #     apinum = getInformation(url, apinum, save_path + "internalHelpers/", version, internal)


def getInformation(url, apinum, save_path, version, dir):
    print(url)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    divs = soup.find_all('div', {'class': 'docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8'})
    for div in divs:
        span = div.find('span', {'class': 'group flex space-x-1.5 items-center text-gray-800 bg-gradient-to-r rounded-tr-lg -mt-4 -ml-4 pt-3 px-2.5'})
        apinum += 1
        apiName = span['id']
        apiTitle = span.text
        if 'class' in apiTitle:
            apiType = 'class'
        else:
            apiType = 'function'
        print(apinum, apiName, apiType)
        p = div.find('p', {'class', 'font-mono text-xs md:text-sm !leading-relaxed !my-6'})
        args = p.text.strip()
        Returns = ''
        if '→' in args:
            Returns = args.split('→')[1].strip()
            args = args.split('→')[0]
        args = re.sub(r'^\(|\)$', '', args)
        args = args.split()
        args_default = {}
        args_delete = []
        for i in range(1, len(args)):
            if args[i] == '=':
                args_delete.append(args[i])
                if i < len(args) - 1:
                    args_default[args[i-1]] = args[i + 1]
                    args_delete.append(args[i + 1])
            if ':' in args[i]:
                args[i] = args[i].replace(':', '')
                args_delete.append(args[i + 1])
        args_list = [x for x in args if x not in args_delete]
        divParams = div.find('div', {'class': '!mb-10 relative docstring-details'})
        lis = divParams.find_all('li', {'class': 'text-base !pl-4 my-3 rounded'})
        params = []
        for li in lis:
            temp = li.text.replace('\n', '')
            name = li.find('strong').text
            temp = temp[len(name) + 1:]
            type = temp.split('—')[0].replace('\n', '')
            description = temp[len(type) + 1:].replace('\n', '').replace('\t', '').replace('"', '').replace(';', ',').strip()
            default = ''
            if name in list(args_default.keys()):
                default = args_default[name]
            optional = False
            if 'optional' in type:
                type = type.split(', optional')[0]
                optional = True
            if 'or' in type:
                type = type.replace(' or ', ',')
            type = type.strip()
            type = re.sub(r'^\(|\)$', '', type)
            params.append((name, type, optional, default, description))
        path = save_path + dir + '/'
        jsonDumps(apiName, apiType, args_list, params, Returns, path, version)
    
    return apinum


def jsonDumps(api, apiType, args_list, params, Returns, path, version):
    jsDict = {}
    myParams = []
    for i in params:
        temp = {}
        temp['name'] = i[0]
        temp['type'] = i[1]
        temp['optional'] = i[2]
        temp['default'] = dealDefault(i[3])
        temp['description'] = dealDefault(i[4])
        myParams.append(temp)
    jsDict['api'] = api
    jsDict['type'] = apiType
    jsDict['version'] = version
    jsDict['args_list'] = args_list
    jsDict['params'] = myParams
    jsDict['return'] = Returns
    fileName = api + '.json'
    filePath = path + fileName
    if os.path.exists(filePath):
        filePath = filePath.replace(".json", "_2.json")
    if not os.path.exists(path):
        os.makedirs(path)
    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(jsDict, f, ensure_ascii=False, indent=4, default=str)
        f.close()

def main():
    # 爬取 Transformers 算子信息
    version = 'main'
    root = "https://huggingface.co/docs/transformers/" + version + "/en/"
    save_path = "node/Transformers-"+ version + "/"
    solve(root, save_path, version)
    

if __name__ == '__main__':
    main()

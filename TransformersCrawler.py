from bs4 import BeautifulSoup
import re, os, json, requests
from Tools import *
from selenium import webdriver

mainClasses = ['agent', 'auto', 'backbones', 'callbacks', 'configuration', 
               'data_collator', 'keras_callbacks', 'logging', 'models', 
               'text_generation', 'onnx', 'optimization', 'output', 'pipelines', 
               'processors', 'quantization', 'tokenizer', 'trainer', 'deepspeed', 
               'feature_extractor', 'image_processor']
# mainClasses = []
models = []
models = []
internalHelpers = ['modeling_utils', 'pipelines_utils', 'tokenization_utils', 
                   'trainer_utils', 'generation_utils', 'image_processing_utils', 
                   'audio_utils', 'file_utils', 'time_series_utils']
# internalHelpers = []
def solve(root, save_path, version):
    apinum = 0
    driver = webdriver.Chrome()
    url = root + "index"
    driver.get(url)
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    trs = []
    if tbody:
        trs = tbody.find_all('tr')
    for td in trs:
        if td.find('a'):
            models.append(td.find('a')['href'])
    for href in models[14:]:
        url = root + href
        apinum = getInformation(url, apinum, save_path, version, href)
    
    for Class in mainClasses:
        url = root + "main_classes/" + Class
        apinum = getInformation(url, apinum, save_path + "main_classes/", version, Class)

    for internal in internalHelpers:
        url = root + "internal/" + internal
        apinum = getInformation(url, apinum, save_path + "internalHelpers/", version, internal)


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
# '''
# python Crawler.py
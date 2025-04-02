from bs4 import BeautifulSoup
import os, json, requests
from Tools import *

apis = ['paddlenlp.data', 'paddlenlp.datasets', 'paddlenlp.embeddings', 
        'paddlenlp.layers', 'paddlenlp.losses', 'paddlenlp.metrics', 
        'paddlenlp.ops', 'paddlenlp.seq2vec', 'paddlenlp.utils']
# apis = ['paddlenlp.data']
modules = ['paddlenlp.taskflow', 'paddlenlp.trainer', 'paddlenlp.transformers']
# modules = []
def solve(root, save_path, version):
    apinum = 0
    for api in apis:
        url = root + api + ".html"
        r = requests.get(url)
        links = getRef(r.text)
        for link in links:
            url = root + link
            apinum = getInformation(url, apinum, save_path, version, api)
        
    for module in modules:
        url = root + module + ".html"
        r = requests.get(url)
        links = getRef(r.text)
        for link in links:
            url = root + link
            apinum = getInformation(url, apinum, save_path, version, module)
        

def getRef(req):
    soup = BeautifulSoup(req, 'html.parser')
    divs = soup.find_all('div', {'class': 'toctree-wrapper compound'})
    lis = []
    for div in divs:
        if div.find_all('li', {'class': 'toctree-l1'}):
            lis += div.find_all('li', {'class': 'toctree-l1'})    
    links = []
    for li in lis:
        if li.find('a', class_='reference internal'):
            links.append(li.find('a', class_='reference internal')['href'])
    return links
   

def getInformation(url, apinum, save_path, version, dir):
    print(url)
    path = save_path + dir + "/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    dls_class = soup.find_all('dl', {'class': 'py class'})
    for dl_class in dls_class:
        apinum += 1
        getClass(dl_class, apinum, path, version)
        dls_method = dl_class.find_all('dl', {'class': 'py method'})
        for dl_method in dls_method:
            apinum += 1
            getMethod(dl_method, apinum, path, version)
    dls_function = soup.find_all('dl', {'class': 'py function'})
    for dl_function in dls_function:
        apinum += 1
        getFunction(dl_function, apinum, path, version)
    
    return apinum


def getFunction(dl_function, apinum, path, version):
    dt = dl_function.find('dt', {'class': 'sig sig-object py'})
    apiName = dt['id']
    apiType = 'function'
    print(apinum, apiName, apiType)
    args_list = {}
    args_default = {}
    ems = dt.find_all('em', {'class': 'sig-param'})
    args = ""
    for em in ems:
        spans = em.find_all('span', {'class': 'pre'})
        for span in spans:
            args += span.text
        args += ","
    args = args.split(',')
    args = args[:-1]
    for i in range(len(args)):
        if ':' in args[i]:
            args_list[args[i].split(':')[0]] = args[i].split(':')[1]
            args_default[args[i].split(':')[0]] = args[i].split(':')[1]
        elif '=' in args[i]:
            args_list[args[i].split('=')[0]] = args[i].split('=')[1]
            args_default[args[i].split('=')[0]] = args[i].split('=')[1]
        else:
            args_list[args[i]] = None
    Returns = ["", ""]
    span = dt.find('span', {'class': 'sig-return-typehint'})
    if span: Returns[1] = span.text
    dd = dl_function.find('dd')
    ps = dd.find_all('p')
    desc = ""
    for p in ps:
        desc += p.text
    desc = desc.replace('\n', '').strip()
    params, Returns = getParams_Returns(dl_function, args_default)
    jsonDumps(apiName, apiType, args_list, None, desc, params, Returns, path, version)


def getClass(dl_class, apinum, path, version):
    args_default = {}
    dt = dl_class.find('dt', {'class': 'sig sig-object py'})
    apiName = dt['id']
    apiType = 'class'
    print(apinum, apiName, apiType)
    args = ""
    args_list = {}
    ems = dt.find_all('em', {'class': 'sig-param'})
    for em in ems:
        spans = em.find_all('span', {'class': 'pre'})
        for span in spans:
            args += span.text
        args += ","
    args = args.split(',')
    args = args[:-1]
    for i in range(len(args)):
        if ':' in args[i]:
            args_list[args[i].split(':')[0]] = args[i].split(':')[1].replace('|', ' or ').replace('=', ' = ')
            continue
        if '=' in args[i]:
            args_default[args[i].split('=')[0]] = args[i].split('=')[1]
            args_list[args[i].split('=')[0]] = args[i].split('=')[1]
        else:
            args_list[args[i]] = None
    dd = dl_class.find('dd')
    ps = dd.find_all('p')
    bases = ""
    desc = ""
    if 'Bases:' in ps[0].text:
        bases = ps[0].text.split(': ')[1]
        if len(ps) > 1: desc = ps[1].text.replace('\n', '').strip()
    else:
        desc = ps[0].text.replace('\n', '').strip()
    params, Returns = getParams_Returns(dl_class, args_default)
    jsonDumps(apiName, apiType, args_list, bases, desc, params, Returns, path, version)


def getParams_Returns(dl, args_default):
    params = []
    Returns = ["", ""]
    if dl.find('dl', class_ = "field-list simple"):
        dts = dl.find('dl', class_ = "field-list simple").find_all('dt')
        for dt in dts:
            if 'Parameters' in dt:
                ps = []
                if dt.parent.find('dd'):
                    ps = dt.parent.find('dd').find_all('p')
                pas = []
                pa = []
                for i in range(len(ps)):
                    if ps[i].strong:
                        pas.append(i)
                    if not ps[i].strong:
                        pa.append(i - 1)
                for i in range(len(pas)):
                    p = ps[pas[i]]
                    name = p.strong.text
                    temp = p.text
                    if pas[i] in pa:
                        j = pas[i]
                        while True:
                            j = j + 1                            
                            if j not in pa:
                                break
                            temp += ps[j].text                    
                    temp = temp[len(name):]
                    type = temp.split('–')[0]
                    description = temp.split('–')[1]
                    type = type.replace('\n', '').replace(')', '').replace('(', '').strip()
                    description = description.replace('\n', '').replace('\t', '').replace('"', '').replace(';', ',').strip()
                    if '|' in type:
                        type = type.replace('|', ' or ')
                    optional = False
                    if 'optional' in type:
                        optional = True
                        type = type.split(', optional')[0]
                    default = ""
                    if name in args_default.keys():
                        default = args_default[name]
                    params.append((name, type, description, default, optional))
                if dt.parent.find('dd'): 
                    dt.parent.find('dd').extract()
            elif 'Returns' in dt:
                ps = []
                if dt.parent.find('dd'):
                    ps = dt.parent.find('dd').find_all('p')
                for p in ps:
                    Returns[0] += p.text
                Returns[0] = Returns[0].replace('\n', '').strip()
                if dt.parent.find('dd'):
                    dt.parent.find('dd').extract()
            elif 'Return type' in dt:
                dd = dt.parent.find('dd')
                if dd:
                    p = dd.find('p')
                    Returns[1] = p.text
                if '|' in Returns[1]:
                    Returns[1] = Returns[1].replace('|', ' or ')
    return params, Returns


def getMethod(dl_method, apinum, path, version):
    dt = dl_method.find('dt', {'class': 'sig sig-object py'})
    apiName = dt['id']
    apiType = 'method'
    print(apinum, apiName, apiType)
    args_list = {}
    ems = dt.find_all('em', {'class': 'sig-param'})
    args = ""
    for em in ems:
        spans = em.find_all('span', {'class': 'pre'})
        for span in spans:
            args += span.text
        args += ","
    args = args.split(',')
    args = args[:-1]
    args_default = {}
    for i in range(len(args)):
        if ':' in args[i]:
            dfault = args[i].split(':')[1]
            if '|' in dfault:
                dfault = dfault.replace('|', ' or ')
            if '=' in dfault:
                dfault = dfault.replace('=', ' = ')
            args_list[args[i].split(':')[0]] = dfault
            args_default[args[i].split(':')[0]] = dfault
        elif '=' in args[i]:
            args_list[args[i].split('=')[0]] = args[i].split('=')[1]
            args_default[args[i].split('=')[0]] = args[i].split('=')[1]
        else:
            args_list[args[i]] = None
    desc = ""   
    params, Returns = getParams_Returns(dl_method, args_default)
    jsonDumps(apiName, apiType, args_list, None, desc, params, Returns, path, version)


def jsonDumps(api, apiType, args_list, bases, desc, params, Returns, path, version):
    jsDict = {}
    myParams = []
    myReturns = {'description': Returns[0], 'type': Returns[1]}
    for i in params:
        temp = {}
        temp['name'] = i[0]
        temp['type'] = i[1]
        temp['description'] = dealDefault(i[2])
        temp['default'] = dealDefault(i[3])
        temp['optional'] = i[4]            
        myParams.append(temp)
    jsDict['api'] = api
    jsDict['type'] = apiType
    jsDict['version'] = version
    jsDict['args_list'] = args_list
    if apiType == 'class': jsDict['Bases'] = bases
    jsDict['description'] = desc
    jsDict['params'] = myParams
    jsDict['return'] = myReturns
    fileName = api + '.json'
    filePath = path + fileName
    if os.path.exists(filePath):
        filePath = filePath.replace(".json", "_2.json")
    if not os.path.exists(path):
        os.makedirs(path)
    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(jsDict, f, ensure_ascii=False, indent=4, default=str)
        # print(f)
        f.close()


def main():
    version = 'stable'
    root = "https://paddlenlp.readthedocs.io/en/" + version + "/source/"
    save_path = "node/" + version + "/"
    solve(root, save_path, version)
    

if __name__ == '__main__':
    main()

# python PaddleNLPCrawler.py
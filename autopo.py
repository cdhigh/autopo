#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""使用ai自动翻译po文件
"""
import os, sys, re, json, argparse, time, datetime, shutil
import polib
import ai_providers

__Version__ = '1.0'
appDir = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON = os.path.join(appDir, 'config.json')
BATCH_SIZE = 2000   #每次翻译的字节数量

SYS_PROMPT = """You are a renowned translation expert, translate the text in a professional and elegant manner without sounding like a machine translation.

Your primary focus is to deliver translations following these guidelines:

- Return in a valid JSON dictionary format only, with no extra content.
- Only translate the text and not interpret it further.
- Do not translate format placeholders like {}, {0}, {name}, %s, %(name)s, etc.
- Do not translate HTML tag names like <br/>.
- The translation should match the original length as closely as possible, being shorter if needed, but not significantly longer.
- Focus solely on delivering precise, concise, friendly, semantically accurate translations.
- Include minimal origin language words, only those that are untranslatable."""

TR_PROMPT = """I will provide a JSON dictionary below. Please translate the keys from {src} to {dst} and use the translations as their corresponding values.
Return the fully translated JSON dictionary in the same structure, without any explanations or additional comments.

JSON dictionary:
{text}
"""

TR_REF_PROMPT = """I will provide a JSON dictionary below. Please translate the keys from {src} to {dst} and use the translations as their corresponding values. 
The original values (if no empty) in the dictionary are {refLang} translations of the keys, provided as a reference to help you translate them more accurately. Replace the values with your translations.
Return the fully translated JSON dictionary in the same structure, without any explanations or additional comments.

JSON dictionary:
{text}"""

LANGUAGE_CODES = {
    "en": "English",
    "zh": "Chinese",
    "zh_cn": "Simplified Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ms": "Malay",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "fa": "Persian",
    "sw": "Swahili",
    "ta": "Tamil",
    "ur": "Urdu",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "id": "Indonesian",
    "th": "Thai",
    "he": "Hebrew",
    "el": "Greek",
    "cs": "Czech",
    "sv": "Swedish",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
}

#在这个列表中的文本不翻译，直接使用原来的文本
EXCLUDED_LIST = ['.', '...']

#创建AI对象
#cfgFile: 配置文件，json格式
def createAiAgent(cfgFile=None):
    cfgFile = cfgFile or CONFIG_JSON
    with open(cfgFile, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    name = cfg.get('provider')
    model = cfg.get('model')
    apiKey = cfg.get('api_key')
    apiHost = cfg.get('api_host')
    chatType = cfg.get('chat_type')
    if not all([name, model, apiKey]):
        raise ValueError('Some parameter is missing')
    singleTurn = bool(chatType == 'single_turn')
    return ai_providers.SimpleAiProvider(name=name, model=model, apiKey=apiKey, apiHost=apiHost, 
        singleTurn=singleTurn)

#翻译一个po文件，保存为同一个文件
#fileName: 需要翻译的po文件
#agent: SimpleAiProvider实例
#dstLang/srcLang: 目标语言代码/源语言
#refPoFile: 需要用作参考的已经手工翻译的其他语种的po文件，可以让AI更准确的翻译
#refLang: 参考po文件的语种
#fuzzify: 是否标识刚翻译的词条为fuzzy
#excluded: 需要排除的翻译文本列表
def translateFile(fileName, agent, dstLang, srcLang=None, refPoFile='', refLang=None, fuzzify=False, excluded=None):
    print(f'Translating to {LANGUAGE_CODES.get(dstLang, dstLang)} by {str(agent)}')
    srcLang = srcLang or 'en'
    excluded = (excluded or []) + EXCLUDED_LIST
    refTrDic = {}
    if refPoFile and refLang: #参考翻译
        refPo = polib.pofile(refPoFile)
        refTrDic = dict([(e.msgid, e.msgstr) for e in refPo.translated_entries() if e.msgid and e.msgstr])

    objDic = {} #待翻译字符串和entry对象的对应关系
    po = polib.pofile(fileName)
    entries = po.untranslated_entries() + po.fuzzy_entries()
    objDic = dict([(e.msgid, e) for e in entries if e.msgid])

    #开始翻译
    toTr = {key: refTrDic.get(key, '') for key in objDic}
    batch = {}
    currLen = 0
    totalCnt = 0
    for key, value in toTr.items():
        if key in excluded:
            objDic[key].msgstr = key
            objDic[key].fuzzy = fuzzify
            totalCnt += 1
            continue

        batch[key] = value
        currLen += len(key) + len(value)
        if currLen > BATCH_SIZE:
            cnt = translateBatch(agent, batch, dstLang, srcLang, refLang, objDic, fuzzify)
            batch = {}
            currLen = 0
            if cnt:
                totalCnt += cnt
            else:
                break

    #剩余部分
    if batch:
        cnt = translateBatch(agent, batch, dstLang, srcLang, refLang, objDic, fuzzify)
        if cnt:
            totalCnt += cnt

    if totalCnt:
        po.save(fileName)
        po = polib.pofile(fileName) #重新读取一次

    print('Translation finised.')
    print(f'Number of translated: {totalCnt}, percent of translated: {po.percent_translated()}%')

#翻译某一批次的文本
#agent: SimpleAiProvider实例
#batch: 要翻译的字典，键为待翻译字符串
#dstLang/srcLang: 目标语言代码/源语言
#refLang: 参考翻译文本的语种，如果存在的话
#objDic: 键对应到entry实例的字典
#fuzzify: 是否标识刚翻译的词条为fuzzy
#返回已经翻译的条目数量
def translateBatch(agent, batch, dstLang, srcLang, refLang, objDic, fuzzify=False):
    print(f'  Translating a batch: {len(batch)}')
    msg = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": ''}]
    text = json.dumps(batch, separators=(',', ':'), ensure_ascii=False)
    src = LANGUAGE_CODES.get(srcLang, srcLang)
    dst = LANGUAGE_CODES.get(dstLang, dstLang)
    refLang = LANGUAGE_CODES.get(refLang, refLang)
    if refLang:
        msg[1]['content'] = TR_REF_PROMPT.format(text=text, src=src, dst=dst, refLang=refLang)
    else:
        msg[1]['content'] = TR_PROMPT.format(text=text, src=src, dst=dst)
    #print(msg) #TODO
    interval = (60 / agent.rpm) if (agent.rpm > 0) else 10 #两次请求直接的间隔
    try:
        respTxt, host = agent.chat(msg)
        time.sleep(interval)
    except Exception as e:
        print(f'Error: {str(e)}, retrying')
        time.sleep(interval + 30)
        try:
            respTxt, host = agent.chat(msg) #再失败就直接退出
            time.sleep(interval)
        except Exception as e:
            print(f'Error again: {str(e)}, breaking')
            return 0

    if not respTxt:
        print('Response is empty, breaking')
        return 0

    #处理这一批次的翻译结果
    #print(respTxt) #TODO
    #清理返回结果的markdown标识
    respTxt = respTxt.strip('` \n\t')
    if respTxt.startswith('json'):
        respTxt = respTxt[4:].strip()

    try:
        ret = json.loads(respTxt)
    except:
        print('Received json is invalid')
        return 0

    cnt = 0
    for k, v in ret.items():
        if not k:
            print('Found a empty key')
            continue
        elif not v:
            print(f'Found a empty value for key in translated: {k}')
            continue
        elif entry := objDic.get(k):
            entry.msgstr = v
            entry.fuzzy = fuzzify
            cnt += 1
        else:
            print(f'The key in translated is modified? {k}')
    return cnt

#分析命令行参数
def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Specify the po file for translation")
    parser.add_argument("-d", "--dest", metavar="LANG", help="Specify the target language", required=True)
    parser.add_argument("-s", "--src", metavar="LANG", help="Specify the source language")
    parser.add_argument("-p", "--refpo", metavar="FILE", help="Specify a reference po file")
    parser.add_argument("-r", "--reflang", metavar="LANG", help="Specify the reference language")
    parser.add_argument("-c", "--config", metavar="FILE", help="Specify a configuration file")
    return parser.parse_args()

if __name__ == "__main__":
    print('Use AI services to automatically translate PO files.')
    print(f'Version: v{__Version__}')

    args = getArg()
    cfgFile = os.path.abspath(args.config) if args.config else None
    refPo = args.refpo
    refLang = args.reflang
    if bool(refPo) != bool(refLang):
        print('You have to provide both --refpo and --reflang')
        sys.exit(0)

    if refPo:
        refPo = os.path.abspath(refPo)

    agent = createAiAgent(cfgFile)
    translateFile(fileName=args.file, agent=agent, dstLang=args.dest, srcLang=args.src,
        refPoFile=args.refpo, refLang=args.reflang)
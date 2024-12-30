#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#一个AI服务的简单封装，提供一个统一的借口
#Author: cdhigh <https://github.com/cdhigh>
import os, sys, json, ssl
import http.client
from urllib.parse import urlsplit

#支持的AI服务商列表，models里面的第一项请设置为默认要使用的model
#context: 输入上下文长度，因为程序采用估计法，建议设小一些。注意：一般的AI的输出长度较短，大约4k/8k
#rpm(requests per minute)是针对免费用户的，如果是付费用户，一般会高很多，可以自己修改
#大语言模型发展迅速，估计没多久这些数据会全部过时
AI_LIST = {
    'google': {'host': 'https://generativelanguage.googleapis.com', 'models': [
        {'name': 'gemini-1.5-flash', 'rpm': 60, 'context': 128000}, #其实支持100万
        {'name': 'gemini-1.5-flash-8b', 'rpm': 60, 'context': 128000},
        {'name': 'gemini-1.5-pro', 'rpm': 10, 'context': 128000},
        {'name': 'gemini-2.0-flash-exp', 'rpm': 10, 'context': 128000},
        {'name': 'gemini-2.0-flash-thinking-exp', 'rpm': 10, 'context': 128000},],},
    'openai': {'host': 'https://api.openai.com', 'models': [
        {'name': 'gpt-4o-mini', 'rpm': 3, 'context': 128000},
        {'name': 'gpt-4o', 'rpm': 3, 'context': 128000},
        {'name': 'gpt-4-turbo', 'rpm': 3, 'context': 128000},
        {'name': 'gpt-3.5-turbo', 'rpm': 3, 'context': 16000},
        {'name': 'gpt-3.5-turbo-instruct', 'rpm': 3, 'context': 4000},],},
    'anthropic': {'host': 'https://api.anthropic.com', 'models': [
        {'name': 'claude-2', 'rpm': 5, 'context': 100000},
        {'name': 'claude-3', 'rpm': 5, 'context': 200000},
        {'name': 'claude-2.1', 'rpm': 5, 'context': 100000},],},
    'xai': {'host': 'https://api.x.ai', 'models': [
        {'name': 'grok-beta', 'rpm': 60, 'context': 128000},
        {'name': 'grok-2', 'rpm': 60, 'context': 128000},],},
    'mistral': {'host': 'https://api.mistral.ai', 'models': [
        {'name': 'open-mistral-7b', 'rpm': 60, 'context': 32000},
        {'name': 'mistral-small-latest', 'rpm': 60, 'context': 32000},
        {'name': 'open-mixtral-8x7b', 'rpm': 60, 'context': 32000},
        {'name': 'open-mixtral-8x22b', 'rpm': 60, 'context': 64000},
        {'name': 'mistral-medium-latest', 'rpm': 60, 'context': 32000},
        {'name': 'mistral-large-latest', 'rpm': 60, 'context': 128000},
        {'name': 'pixtral-12b-2409', 'rpm': 60, 'context': 128000},],},
    'groq': {'host': 'https://api.groq.com', 'models': [
        {'name': 'gemma2-9b-it', 'rpm': 30, 'context': 8000},
        {'name': 'gemma-7b-it', 'rpm': 30, 'context': 8000},
        {'name': 'llama-guard-3-8b', 'rpm': 30, 'context': 8000},
        {'name': 'llama3-70b-8192', 'rpm': 30, 'context': 8000},
        {'name': 'llama3-8b-8192', 'rpm': 30, 'context': 8000},
        {'name': 'mixtral-8x7b-32768', 'rpm': 30, 'context': 32000},],},
    'perplexity': {'host': 'https://api.perplexity.ai', 'models': [
        {'name': 'llama-3.1-sonar-small-128k-online', 'rpm': 60, 'context': 128000},
        {'name': 'llama-3.1-sonar-large-128k-online', 'rpm': 60, 'context': 128000},
        {'name': 'llama-3.1-sonar-huge-128k-online', 'rpm': 60, 'context': 128000},],},
    'alibaba': {'host': 'https://dashscope.aliyuncs.com', 'models': [
        {'name': 'qwen-turbo', 'rpm': 60, 'context': 128000}, #其实支持100万
        {'name': 'qwen-plus', 'rpm': 60, 'context': 128000},
        {'name': 'qwen-long', 'rpm': 60, 'context': 128000},
        {'name': 'qwen-max', 'rpm': 60, 'context': 32000},],},
}

#自定义HTTP响应错误异常
class HttpResponseError(Exception):
    def __init__(self, status, reason, body=None):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason
        self.body = body

class SimpleAiProvider:
    #name: AI提供商的名字
    #apiKey: 如需要多个Key，以分号分割，逐个使用
    #apiHost: 支持自搭建的API转发服务器，传入以分号分割的地址列表字符串，则逐个使用
    #singleTurn: 一些API转发服务不支持多轮对话模式，设置此标识，当前仅支持 openai
    def __init__(self, name, apiKey, model=None, apiHost=None, singleTurn=False):
        name = name.lower()
        if name not in AI_LIST:
            raise ValueError(f"Unsupported provider: {name}")
        self.name = name
        self.apiKeys = apiKey.split(';')
        self.apiKeyIdx = 0
        self.singleTurn = singleTurn
        self._models = AI_LIST[name]['models']
        
        #如果传入的model不在列表中，默认使用第一个
        item = next((m for m in self._models if m['name'] == model), self._models[0])
        self.model = item['name']
        self.rpm = item['rpm']
        self.context_size = item['context']
        if self.rpm <= 0:
            self.rpm = 2
        if self.context_size < 1000:
            self.context_size = 1000
        #分析主机和url，保存为 SplitResult(scheme,netloc,path,query,frament)元祖
        #connPools每个元素为 [host_tuple, conn_obj]
        self.connPools = [[urlsplit(e if e.startswith('http') else ('https://' + e)), None]
            for e in (apiHost or AI_LIST[name]['host']).replace(' ', '').split(';')]
        self.host = '' #当前正在使用的 netloc
        self.connIdx = 0
        self.createConnections()

    #自动获取下一个ApiKey
    @property
    def apiKey(self):
        ret = self.apiKeys[self.apiKeyIdx]
        self.apiKeyIdx = (self.apiKeyIdx + 1) % len(self.apiKeys)
        return ret

    #自动获取列表中下一个连接对象，返回 (index, host tuple, con obj)
    def nextConnection(self):
        index = self.connIdx
        host, conn = self.connPools[index]
        self.connIdx = (self.connIdx + 1) % len(self.connPools)
        return index, host, conn

    #创建长连接
    #index: 如果传入一个整型，则只重新创建此索引的连接实例
    def createConnections(self):
        for index in range(len(self.connPools)):
            self.createOneConnection(index)

        #尽量不修改connIdx，保证能轮询每个host
        if self.connIdx >= len(self.connPools):
            self.connIdx = 0

    #创建一个对应索引的连接对象
    def createOneConnection(self, index):
        if not (0 <= index < len(self.connPools)):
            return

        host, e = self.connPools[index]
        if e:
            e.close()
        #使用http.client.HTTPSConnection有一个好处是短时间多次对话只需要一次握手
        if host.netloc.endswith('duckduckgo.com'):
            conn = DuckOpenAi()
        elif host.scheme == 'https':
            sslCtx = ssl._create_unverified_context()
            conn = http.client.HTTPSConnection(host.netloc, timeout=60, context=sslCtx)
        else:
            conn = http.client.HTTPConnection(host.netloc, timeout=60)
        self.connPools[index][1] = conn

    #发起一个网络请求，返回json数据
    def _send(self, path, headers=None, payload=None, toJson=True, method='POST') -> dict:
        if payload:
            payload = json.dumps(payload)
        retried = 0
        while retried < 2:
            try:
                index, host, conn = self.nextConnection() #(index, host_tuple, conn_obj)
                self.host = host.netloc
                #拼接路径，避免一些边界条件出错
                url = '/' + host.path.strip('/') + (('?' + host.query) if host.query else '') + path.lstrip('/')
                conn.request(method, url, payload, headers)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8")
                #print(resp.reason, ', ', body) #TODO
                print(url)
                if not (200 <= resp.status < 300):
                    raise HttpResponseError(resp.status, resp.reason, body)
                return json.loads(body) if toJson else body
            except (http.client.CannotSendRequest, http.client.RemoteDisconnected) as e:
                if retried:
                    raise
                #print("Connection issue, retrying:", e)
                self.createOneConnection(index)
                retried += 1

    #关闭连接
    #index: 如果传入一个整型，则只关闭对应索引的连接
    def close(self, index=None):
        connNum = len(self.connPools)
        if isinstance(index, int) and (0 <= index < connNum):
            host, e = self.connPools[index]
            if e:
                e.close()
                self.connPools[index][1] = None

        for index in range(connNum):
            host, e = self.connPools[index] #[host_tuple, conn_obj]
            if e:
                e.close()
                self.connPools[index][1] = None

    def __repr__(self):
        return f'{self.name}/{self.model}'

    #外部调用此函数即可调用简单聊天功能
    #message: 如果是文本，则使用各项默认参数
    #传入 list/dict 可以定制 role 等参数
    #返回 respTxt，如果要获取当前使用的主机，可以使用 host 属性
    def chat(self, message) -> (str, str):
        if not self.apiKey:
            raise ValueError(f'The api key is empty')
        name = self.name
        if name == "openai":
            return self._openai_chat(message)
        elif name == "anthropic":
            return self._anthropic_chat(message)
        elif name == "google":
            return self._google_chat(message)
        elif name == "xai":
            return self._xai_chat(message)
        elif name == "mistral":
            return self._mistral_chat(message)
        elif name == 'groq':
            return self._groq_chat(message)
        elif name == 'perplexity':
            return self._perplexity_chat(message)
        elif name == "alibaba":
            return self._alibaba_chat(message)
        else:
            raise ValueError(f"Unsupported provider: {name}")

    #返回当前服务提供商支持的models列表
    def models(self, prebuild=True):
        if self.name in ('openai', 'xai'):
            return self._openai_models()
        elif self.name == 'google':
            return self._google_models()
        else:
            return [item['name'] for item in self._models]

    #openai的chat接口
    def _openai_chat(self, message, path='v1/chat/completions'):
        headers = {'Authorization': f'Bearer {self.apiKey}', 'Content-Type': 'application/json'}
        if isinstance(message, str):
            msg = [{"role": "user", "content": message}]
        elif self.singleTurn and (len(message) > 1): #将多轮对话手动拼接为单一轮对话
            msgArr = ['Previous conversions:\n']
            roleMap = {'system': 'background', 'assistant': 'Your responsed'}
            msgArr.extend([f'{roleMap.get(e["role"], "I asked")}:\n{e["content"]}\n' for e in message[:-1]])
            msgArr.append(f'\nPlease continue this conversation based on the previous information:\n')
            msgArr.append("I ask:")
            msgArr.append(message[-1]['content'])
            msgArr.append("You Response:\n")
            msg = [{"role": "user", "content": '\n'.join(msgArr)}]
        else:
            msg = message
        payload = {"model": self.model, "messages": msg}
        data = self._send(path, headers=headers, payload=payload, method='POST')
        return data["choices"][0]["message"]["content"]

    #openai的models接口
    def _openai_models(self):
        headers = {'Authorization': f'Bearer {self.apiKey}', 'Content-Type': 'application/json'}
        data = self._send('v1/models', headers=headers, payload=None, method='GET')
        return [item['id'] for item in data['data']]

    #anthropic的chat接口
    def _anthropic_chat(self, message):
        headers = {'Accept': 'application/json', 'Anthropic-Version': '2023-06-01',
            'Content-Type': 'application/json', 'x-api-key': self.apiKey}

        if isinstance(message, list): #将openai的payload格式转换为anthropic的格式
            msg = []
            for item in message:
                role = 'Human' if (item.get('role') != 'assistant') else 'Assistant'
                content = item.get('content', '')
                msg.append(f"\n\n{role}: {content}")
            prompt = ''.join(msg) + "\n\nAssistant:"
            payload = {"prompt": prompt, "model": self.model, "max_tokens_to_sample": 256}
        elif isinstance(message, dict):
            payload = message
        else:
            prompt = f"\n\nHuman: {message}\n\nAssistant:"
            payload = {"prompt": prompt, "model": self.model, "max_tokens_to_sample": 256}
        
        data = self._send('v1/complete', payload=payload, headers=headers, method='POST')
        return data["completion"]

    #google的chat接口
    def _google_chat(self, message):
        url = f'v1beta/models/{self.model}:generateContent?key={self.apiKey}'
        headers = {'Content-Type': 'application/json'}
        if isinstance(message, list): #将openai的payload格式转换为gemini的格式
            msg = []
            for item in message:
                role = 'user' if (item.get('role') != 'assistant') else 'model'
                content = item.get('content', '')
                msg.append({'role': role, 'parts': [{'text': content}]})
            payload = {'contents': msg}
        elif isinstance(message, dict):
            payload = message
        else:
            payload = {'contents': [{'role': 'user', 'parts': [{'text': message}]}]}
        data = self._send(url, payload=payload, headers=headers, method='POST')
        contents = data["candidates"][0]["content"]
        return contents['parts'][0]['text']

    #google的models接口
    def _google_models(self):
        url = f'v1beta/models?key={self.apiKey}&pageSize=100'
        headers = {'Content-Type': 'application/json'}
        data = self._send(url, payload=None, headers=headers, method='GET')
        _trim = lambda x: x[7:] if x.startswith('models/') else x
        return [_trim(item['name']) for item in data['models']]

    #xai的chat接口
    def _xai_chat(self, message):
        return self._openai_chat(message, path='v1/chat/completions')

    #mistral的chat接口
    def _mistral_chat(self, message):
        return self._openai_chat(message, path='v1/chat/completions')

    #groq的chat接口
    def _groq_chat(self, message):
        return self._openai_chat(message, path='openai/v1/chat/completions')

    #perplexity的chat接口
    def _perplexity_chat(self, message):
        return self._openai_chat(message, path='chat/completions')

    #通义千问
    def _alibaba_chat(self, message):
        return self._openai_chat(message, path='compatible-mode/v1/chat/completions')

#duckduckgo转openai格式的封装器，外部接口兼容http.HTTPConnection
class DuckOpenAi:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Accept": "text/event-stream",
        "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://duckduckgo.com/",
        "Content-Type": "application/json",
        "Origin": "https://duckduckgo.com",
        "Connection": "keep-alive",
        "Cookie": "dcm=1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "TE": "trailers",
    }
    HOST = "duckduckgo.com"
    STATUS_URL = "/duckchat/v1/status"
    CHAT_URL = "/duckchat/v1/chat"

    #模拟HTTPConnection实例 getresponse() 返回的的结果
    class DuckResponse:
        def __init__(self, status, headers, data, reason=''):
            self.status = status
            self.headers = headers
            self.data = data
            self.reason = reason
        def read(self):
            return self.data

    def __init__(self):
        self.conn = None
        self._payload = {}
        self.createConnection()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def createConnection(self):
        self.close()
        sslCtx = ssl._create_unverified_context()
        self.conn = http.client.HTTPSConnection('duckduckgo.com', timeout=60, context=sslCtx)
        return self.conn

    #使用底层接口实际发送网络请求
    #返回元祖 (status, headers, body)
    def _send(self, url, headers=None, payload=None, method='GET'):
        retried = 0
        _headers = self.HEADERS
        _headers.update(headers)
        while retried < 2:
            try:
                self.conn.request(method, url, payload, _headers)
                resp = self.conn.getresponse()
                return resp.status, resp.headers, resp.read()
            except (http.client.CannotSendRequest, http.client.RemoteDisconnected) as e:
                if retried:
                    raise
                #print("Connection issue, retrying:", e)
                self.createConnection()
                retried += 1
        return 500, {}, b''

    #只是暂存结果，在 getresponse() 才实际发起请求
    def request(self, method, url, payload=None, headers=None):
        self._payload = json.loads(payload or '{}')

    #发起请求，返回 DuckResponse 实例
    def getresponse(self):
        status, heads, body = self._send(self.STATUS_URL, headers={"x-vqd-accept": "1"})
        if status != 200:
            return self.DuckResponse(status, heads, body)
            
        vqd4 = heads.get("x-vqd-4", '')
        payload = {"model": "gpt-4o-mini", "messages": self._payload.get('messages', [])}

        status, heads, body = self._send(self.CHAT_URL, headers={"x-vqd-4": vqd4}, 
            payload=json.dumps(payload), method='POST')
        if status != 200:
            return self.DuckResponse(status, heads, body)

        content = id_ = model = ""
        created = 0
        for line in body.decode('utf-8').splitlines():
            if line.startswith("data: "):
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    id_ = data.get("id", id_)
                    created = data.get("created", created)
                    model = data.get("model", model)
                    content += data.get("message", "")
                except json.JSONDecodeError:
                    continue
        body = {"id": id_, "object": "chat.completion", "created": created, "model": model,
            "choices": [{ "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},},],}
        return self.DuckResponse(status, heads, json.dumps(body).encode('utf-8'))

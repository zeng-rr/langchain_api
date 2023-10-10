from flask import Flask, request

from langchain_service import LangChainService
import config

app = Flask(__name__)

success = {'code': 0, 'msg': 'ok', 'data': {}}
fail = {'code': 1, 'msg': 'params error', 'data': {}}

# 在每个请求到达视图函数之前执行该函数
@app.before_request
def check_api_key():
    data = request.get_json()
    if 'user_id' not in data:
        return fail
    app.service = LangChainService(data['user_id'])

@app.route('/', methods=['GET', 'POST'])
def index():
    return success

# 获取索引
@app.route('/get', methods=['GET', 'POST'])
def get():
    data = request.get_json()
    if 'ids' not in data:
        return fail
    response = app.service.get(data['ids'], data['metadatas'] if 'metadatas' in data else None)
    success['data'] = response
    return success

# 查询问题
@app.route('/query', methods=['GET', 'POST'])
def query():
    data = request.get_json()
    if 'text' not in data:
        return fail
    response = app.service.query(data['text'], data['max_distance'] if 'max_distance' in data else 0.5)
    success['data'] = response
    return success

# 添加索引
@app.route('/add', methods=['POST'])
def add():
    data = request.get_json()
    if 'texts' not in data:
        return fail
    response = app.service.add(data['texts'], data['metadatas'] if 'metadatas' in data else None)
    success['data'] = response
    return success

# 更新索引
@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    if 'ids' not in data or 'texts' not in data:
        return fail
    response = app.service.update(data['ids'], data['texts'], data['metadatas'] if 'metadatas' in data else None)
    success['data'] = response
    return success

# 删除索引
@app.route('/delete', methods=['POST'])
def delete_user():
    data = request.get_json()
    if 'ids' not in data:
        return fail
    response = app.service.delete(data['ids'], data['metadatas'] if 'metadatas' in data else None)
    success['data'] = response
    return success

if __name__ == '__main__':
    app.run(host=config.host, debug=config.debug, port=config.port)
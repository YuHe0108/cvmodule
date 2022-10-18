import logging
from flask import request, Flask

from FLASK.ai_interface import ai_configuration

LOG = logging.getLogger("Flask Interface")


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        if request.method == 'OPTIONS':
            response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
            headers = request.headers.get('Access-Control-Request-Headers')
            if headers:
                response.headers['Access-Control-Allow-Headers'] = headers
        return response

    # 注册路由
    app.register_blueprint(ai_configuration, url_prefix='/ai_interface')

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/<path:filename>')
    def js(filename):
        return app.send_static_file("{}".format(filename))

    @app.route('/<path:filename>.css')
    def css(filename):
        return app.send_static_file("{}.css".format(filename))

    @app.route('/<path:filename>.png')
    def png(filename):
        return app.send_static_file("{}.png".format(filename))

    return app


if __name__ == '__main__':
    app_ = create_app()
    app_.run(port=6666, host="0.0.0.0")
    LOG.info("启动web服务器")

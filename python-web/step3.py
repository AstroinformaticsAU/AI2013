
#!/usr/bin/env python

from bottle import route, run, request, static_file

@route('/css/<filename>')
def server_css(filename):
    """Serve css file from the static directory"""
    return static_file(filename, root='./static/css')

@route('/')
def index():
    """Serve a static html file"""
    return static_file('step3.html', root='./static')

@route('/greet/<name>')
def greet(name):
    valid = ['upper', 'lower', 'title']
    mode = request.query.get('mode', 'title')
    mode = mode.lower()
    if mode in valid:
        f = getattr(name, mode)
        return "Hello %s!" % f()
    else:
        return "Error"

if __name__ == "__main__":
    run(host='localhost', port=8080, debug=True)

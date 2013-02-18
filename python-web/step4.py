#!/usr/bin/env python

from bottle import route, run, request, template


@route('/greet/<name>')
def greet(name="World"):
    valid = ['upper', 'lower', 'title']
    mode = request.query.get('mode', 'title')
    mode = mode.lower()
    if mode in valid:
        f = getattr(name, mode)
        return template('Hello {{name}}, how are you?', name=f())
    else:
        return "Error"


if __name__ == "__main__":
    run(host='localhost', port=8080, debug=True, reloader=True)

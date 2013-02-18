#!/usr/bin/env python

from bottle import route, run

@route('/')
def index():
    return "Hello World!"

@route('/greet/<name>')
def greet(name):
    return "Hello World %s!" % name

if __name__ == "__main__":
    run(host='localhost', port=8080, debug=True)

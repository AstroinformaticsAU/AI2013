#!/usr/bin/env python

from bottle import route, run

@route('/')
def index():
    return "Hello World"

if __name__ == "__main__":
    run(host='localhost', port=8080, debug=True)

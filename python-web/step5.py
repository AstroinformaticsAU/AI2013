#!/usr/bin/env python

import sqlite3
from bottle import route, run, request, template


@route('/columns')
def columns():
    conn = sqlite3.connect('hipass.db')
    c = conn.cursor()
    c.execute("SELECT * FROM hipass")
    # a trick to get column names...
    names = [ desc[0] for desc in c.description ]
    result = c.fetchall()
    result.insert(0, names)
    c.close()
    output = template('step5', rows=result)
    return output


if __name__ == "__main__":
    run(host='localhost', port=8080, debug=True, reloader=True)

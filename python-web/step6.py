#!/usr/bin/env python

import bottle
from bottle import template
import bottle.ext.sqlite

app = bottle.Bottle()
plugin = bottle.ext.sqlite.Plugin(dbfile='hipass.db')
app.install(plugin)

@app.route('/columns')
def columns(db):
    cur = db.execute("SELECT * FROM hipass")
    # a trick to get column names...
    names = [ desc[0] for desc in cur.description ]
    result = cur.fetchall()
    result.insert(0, names)
    output = template('html_table', rows=result)
    return output

@app.route('/source/<oid:int>')
def source(oid, db):
    cur = db.execute("SELECT * FROM hipass WHERE Obj=?", (oid,))
    result = cur.fetchone()
    if len(result) == 0:
        return {'error': 'object not found'}
    names = [ desc[0] for desc in cur.description ]
    return { 'source': dict(zip(names, result)) }



if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True, reloader=True)

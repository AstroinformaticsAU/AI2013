#!/usr/bin/env python

import sqlite3

catalogue = []

# read file and store as lists (rows) of lists (columns)
with open('hipass.txt','r') as f:
    for i, line in enumerate(f):
        #ignore header
        if i == 0:
            continue
        # ignore comments
        line = line.strip()
        if line.startswith("#"):
            continue
        elem = line.split()
        # Obj, Name, RA, DEC
        catalogue.append((elem[0], elem[1], elem[5], elem[6]))

#connection = sqlite3.connect(':memory:')
connection = sqlite3.connect('hipass_min.db')

with connection:
    cur = connection.cursor()    
    cur.execute("DROP TABLE IF EXISTS hipass")
    cur.execute("CREATE TABLE hipass(Id INT, Name TEXT, RA TEXT, DEC TEXT)")
    cur.executemany("INSERT INTO hipass VALUES(?, ?, ?, ?)", catalogue)

    for row in cur.execute("SELECT * FROM hipass WHERE Name LIKE 'J060%' "):
        print row
    print "-" * 20
    cur.execute("SELECT * FROM hipass WHERE Name LIKE 'J061%'")
    result = cur.fetchall()
    for r in result:
        print r
    

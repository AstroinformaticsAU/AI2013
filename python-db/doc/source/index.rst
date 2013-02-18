.. Python sqlite primer documentation master file, created by
   sphinx-quickstart on Mon Feb 18 21:40:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astroinformatics Python sqlite primer
=====================================

Accessing sqlite databases in python is straight forward as a module for
this is built into python::

    import sqlite3

A database connection is setup using :func:`sqlite3.connect` which takes a 
*filename* for storage on disk or *\:memory\:*::

    connection = sqlite3.connect('hipass_min.db')

From the connection a cursor can be retrieved to access datbase records::
  
    cur = connection.cursor()

This cursor is used to execute SQL statements, e.g.

.. literalinclude:: ../../tutorial.py
  :lines: 26-28


To get better performance use :meth:`sqlite3.Cursor.executemany` when lots of 
operations are needed. It takes a row/column list of lists as input.

Finally a cursor can be used as an iterator to step through results or :meth:`sqlite3.Cursor.fetchall` for a list of all rows. Both examples here achive the 
same result:

.. literalinclude:: ../../tutorial.py
  :lines: 30-36
  :language: python


Exercise
--------

1. Add additional columns from the file with different data types than string
2. Modify the example to buffer databse writes 10 input lines at a time. This saves on memory for large input files and is better for error recovery.

References
----------

* `SQLAlchemy <http://www.sqlalchemy.org/>`_ for general DB access

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


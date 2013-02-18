.. Python Web documentation master file, created by
   sphinx-quickstart on Mon Feb 18 14:36:05 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================
Astroinformatics 2013 - Python Web tutorial
===========================================

Background
==========

The Internet
------------

* A network of networks
* The Internet is the descendant of ARPANET (Advanced Research Projects Agency Network) developed for the US DoD
* The initial goal was to research the possibility of remote communication between machines
* Critical step was development of the TCP/IP protocol (1977)

  - TCP Transmission Control Protocol
  - IP Internet Protocol

* Vinton Cerf’s postcard analogy for TCP/IP

  - A document is broken up into postcard-sized chunks (packets)
  - Each postcard has its own address and sequence number
  - Each postcard travels independently to the ﬁnal destination
  - The document is reconstructed by ordering the postcards
  - If one is missing, the recipient can request for it to be resent
  - If a post-oﬃce is closed the postcard is sent a diﬀerent way
  - Congestion and service interruptions do not stop transmission

World Wide Web
--------------

* A service operating over the Internet
* The concept of the WWW combines 4 ideas:

  - Hypertext
  - resource identiﬁers (URI, URL)
  - client-server model of computing (web servers/browsers)
  - markup language (HTML)

* These were the brainchild of Tim Berners-Lee from CERN who released his ﬁrst browser in 1991
* All clients and servers in the WWW speak the language of HTTP (HyperText Transfer Protocol)


Client - Server
---------------

* HyperText Transfer Protocol (http) is the standard protocol for transferring web content
* The server listens on port 80 waiting for connections
* The web browser connects to the server, and sends a request
* The server responds with an error code or the web content
* The server can process many requests at the same time


Dynamic content
---------------
  
* python's cgi module is too low-level, lots of boiler plate code required
* Web frameworks hide this
* Present static and dynamic content

  - html 
  - json or xml
  - other mime-types often dynamically

* Present an API for re-use (mash-ups), e.g. facebook, github etc.
* Rapid prototyping/testing no server needed

Presentation
------------

* HTML content mark-up
* CSS for “presentation” 
  - Can be customised on the client side, e.g. accessibility, mobile devices.
* Client-side modifications - javascript


Let's get going
===============

We will be using a python framework called `bottle <http://bottlepy.org/>`_
This provides 

* templating of HTML code
* easy 'routing' of URLs
* test environment.

Hello World
-----------

The smallest application. It returns text (not HTML) and introduces :func:`bottle.route` and :func:`bottle.run`.

The '@' denotes a python decorator which wraps a function call with commonly
reused code. :func:`bottle.route` will assign a URL to a function call.
:func:`bottle.run` runs this script as a webserver application. Note that 
this is only good for testing. Once the application is to be published it
needs to be deployed in a 'proper' webserver. See the bottle documentation for
this.


Start the program by typing::

    python step1.py

.. literalinclude:: ../../step1.py

Arguments
---------

In this example we dynamically generate output from an variable URL which is
defined by the '<variable>' notation and then passed as an argument to the 
function.

.. literalinclude:: ../../step2.py

Static files and query parameters
---------------------------------

Use :func:`bottle.static_file` to serve static files, such as HTML documents and style sheets (CSS).
Also use query parameters (e.g. myurl?x=0&y=2 )to modify a request dynamically.
This is the first step to handle forms.

.. literalinclude:: ../../step3.py

Basic template
--------------

This introduces :func:`bottle.template` to insert parameters into a reply.


.. literalinclude:: ../../step4.py

HTML templates
--------------

Learn how to insert python code into your HTML document to dynamically modify it. It will generate a table from a sqlite source catalogue database.

.. literalinclude:: ../../step5.py


Access database rows directly from a URL
----------------------------------------

We will use a readable url route to connect to sqlite database rows. This return the rows as a json document which can be used from a client using javascript.
This time a `bottle` sqlite plugin is used to avoid code repition.
We can use the '<variable:type>' notation to automatically convert the variable
to an `int`.


.. literalinclude:: ../../step6.py


Exercise

    Modify the example to allow selection of columns to return. Hint see 'step3.py' and use :func:`bottle.request.getlist`. Multiple arguments can be passed as follows '?x=1&x=2'. 

.. toctree::
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


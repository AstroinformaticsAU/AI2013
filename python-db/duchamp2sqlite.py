# sudo easy_install asciitable atpy
import atpy

tbl = atpy.Table('hipass.txt', type='ascii', name='hipass')
tbl.write('sqlite', 'hipass.db')


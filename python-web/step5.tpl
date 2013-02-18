%#template to generate a HTML table from a list of tuples (or list of lists, or tuple of tuples or ...)
<html>
<body>
<p>The HIPASS table:</p>
<table border="1">
%for row in rows:
  <tr>
  %for col in row:
    <td>{{col}}</td>
  %end
  </tr>
%end
</table>
</body>
</html>
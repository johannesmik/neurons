jQuery(document).ready(function($){

    function epsilon(s, t_current, t_membrane) {
        return (1 / (1 - t_current / t_membrane)) * (Math.exp(-s/t_membrane) - Math.exp(-s/t_current));
      } 
      
      function refreshPlot() {
        var t_current = $( "#current-slider" ).slider( "value" ),
          t_membrane = $( "#membrane-slider" ).slider( "value" );

          var canvas = document.getElementById('epsilon-plot');
          var width = canvas.width;
          var height = canvas.height;
          var mid_x = width / 2;
          var mid_y = height / 2;
          var context = canvas.getContext('2d');
          var margin = 10;

          // Clear canvas
          context.clearRect(0, 0, canvas.width, canvas.height);

          context.font = "12px Arial";

          // Coordinate system
          context.beginPath();
          context.moveTo(margin, height - margin);
          context.lineTo(margin, margin);
          context.moveTo(margin, 140);
          context.lineTo(width - margin, 140);
          context.fillText("eps(s)", 280, 25);
          context.fillText("t-current:" + String(t_current), 280, 40);
          context.fillText("t-membrane:" + String(t_membrane), 280, 55);

          // Coordinate system ticks
          context.moveTo(5, 10);
          context.lineTo(15, 10);
          context.fillText("1", 20, 15);
          for (i = 1; i <= 3; i++) {
            context.moveTo(margin + i * 100, 140 + 5);
            context.lineTo(margin + i * 100, 140 - 5);
            context.fillText(String(i*100), margin + i * 100 + 5, 140 + 10);
          }

          // The function
          context.moveTo(margin, mid_y + 5);
          for (i = 0; i <= 400; i++) {
            var value = epsilon(i, t_current, t_membrane);
            value = ( -value + 1 ) * 130 / 1 + 10;
            context.lineTo(margin + i, value);
          }

          context.stroke();

      }
      $(function() {
        $( "#current-slider" ).slider({
          orientation: "horizontal",
          range: "min",
          max: 201,
          min: 0,
          value: 10,
          slide: refreshPlot,
          change: refreshPlot
        });
        $( "#membrane-slider" ).slider({
          orientation: "horizontal",
          range: "min",
          max: 200,
          min: 0,
          value: 127,
          slide: refreshPlot,
          change: refreshPlot
        });

        $( "#membrane-slider" ).slider("value", 200);

      });
      
});

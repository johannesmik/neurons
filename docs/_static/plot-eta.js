    function eta(s, eta0, t_membrane) {
        return - eta0 * Math.exp(-s/t_membrane);
      } 
      
      function refreshEtaPlot() {
        var eta0 = $( "#eta-eta0-slider" ).slider( "value" ),
          t_membrane = $( "#eta-membrane-slider" ).slider( "value" );
          
          eta0 = eta0 / 10.0;

          var canvas = document.getElementById('eta-plot');
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
          context.moveTo(margin, 10);
          context.lineTo(width - margin, 10);
          context.fillText("eta(s)", 280, 120);
          context.fillText("eta_0: " + String(eta0), 280, 135);
          context.fillText("t-membrane: " + String(t_membrane), 280, 150);

          // Coordinate system ticks
          context.moveTo(5, 140);
          context.lineTo(15, 140);
          context.fillText("-1", 20, 145);
          for (i = 1; i <= 3; i++) {
            context.moveTo(margin + i * 100, 10 + 5);
            context.lineTo(margin + i * 100, 10 - 5);
            context.fillText(String(i*100), margin + i * 100 + 5, 10 + 10);
          }

          // The function
          context.moveTo(margin, mid_y + 5);
          for (i = 0; i <= 400; i++) {
            var value = eta(i, eta0, t_membrane);
            value = ( -value + 1 ) * 130 / 1 - 120;
            context.lineTo(margin + i, value);
          }

          context.stroke();

      }
      $(function() {
        $( "#eta-eta0-slider" ).slider({
          orientation: "horizontal",
          range: "min",
          max: 20,
          min: 0,
          value: 10,
          slide: refreshEtaPlot,
          change: refreshEtaPlot
        });
        $( "#eta-membrane-slider" ).slider({
          orientation: "horizontal",
          range: "min",
          max: 200,
          min: 0,
          value: 60,
          slide: refreshEtaPlot,
          change: refreshEtaPlot
        });

        $( "#eta-membrane-slider" ).slider("value", 100);

      });

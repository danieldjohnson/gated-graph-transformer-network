require.config({
  paths: {
      d3: 'http://cdnjs.cloudflare.com/ajax/libs/d3/4.1.0/d3.min',
      dat: 'http://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.5.1/dat.gui.min'
  }
});

require(['d3','dat'], function(d3,_ignored){
function _graph_display(states,colormap,el,batch,options){
    var node_strengths = states[0];
    var node_ids = states[1];
    var node_states = states[2];
    var edge_strengths = states[3];
    var max_time = node_strengths[batch].length;

    var width = options.width || 500;
    var height = options.height || 500;

    var svg = d3.select(el).append("svg").attr("width",width).attr("height",height);

    if(!options)
        options = {}

    var node_map = {};
    var data_nodes = [];
    var data_edges = [];
    var display_edges = [];
    var selection_map, selection_options;
    var force = d3.forceSimulation()
                    .force("charge", d3.forceManyBody())
                    .force("link", d3.forceLink())
                    .force("gravityX", d3.forceX(width/2))
                    .force("gravityY", d3.forceY(height/2));

    var extra_snap_specs = options.extra_snap_specs || [];
    var extra_forces = [];
    for(var i=0; i<extra_snap_specs.length; i++){
        var spec = extra_snap_specs[i];
        if(spec.axis == "x")
            var new_force = d3.forceX(spec.value + width/2);
        else if(spec.axis == "y")
            var new_force = d3.forceY(spec.value + height/2);
        force.force("extra"+i, new_force);
        extra_forces.push(new_force);
    }

    var edge_strength_adjust = options.edge_strength_adjust
    if(!edge_strength_adjust){
        console.log("Using default edge_strength_adjust")
        edge_strength_adjust = [];
        for(var i=0; i<edge_strengths[0][0][0][0].length; i++)
            edge_strength_adjust.push(1);
    }
    
    var link_fwd = svg.append("g").selectAll("path.fwd");
    var node = svg.append("g").selectAll("circle");
    
    var focus_detail = svg.append("g").selectAll("rect");
    
    var params = {
        linkDistance: 80,
        linkStrength: 0.1,
        gravity: 0.02,
        charge: 100,
        primarySelection: 'Active Selection',
        secondarySelection: 'Stored Selection 0',
        timestep:max_time-1,
    };

    if(options.jitter){
        params.jitterScale = 20;
        function makeJitterForce(){
            var nodes;
            function forcefn(alpha){
                if(alpha<0.2)
                    return;
                var n = nodes.length;
                var node;
                var jitter_amt = alpha * alpha * alpha * params.jitterScale;
                for (var i = 0; i < n; ++i) {
                    node = nodes[i];
                    node.x += jitter_amt * (Math.random()*2-1);
                    node.y += jitter_amt * (Math.random()*2-1);
                }
            }
            forcefn.initialize = function(snodes){
                nodes = snodes;
            }
            return forcefn;
        }

        force.force("jitter", makeJitterForce());
    }

    for(var key in params){
        if(params.hasOwnProperty(key) && options[key] !== undefined)
            params[key] = options[key];
    }
    
    function colfn(key){
        return function(d){
            var c = d[key];
            var col = "rgb("+~~(255*c[0])+","+~~(255*c[1])+","+~~(255*c[2])+")";
            console.log(col);
            return col;
        }
    }
    function poly_d_fn(d) {
      return "M" + d.join("L") + "Z";
    }
    function adjust(template, reverse){
        return function(d){
            if(reverse){
                var target = d.source, source = d.target;
            }else{
                var target = d.target, source = d.source;
            }
            var p_dx = target.x - source.x;
            var p_dy = target.y - source.y;
            var pmag = Math.sqrt(p_dx*p_dx + p_dy*p_dy);
            var s_dx = -p_dy/pmag;
            var s_dy = p_dx/pmag;
            
            var res = [];
            for(var i=0; i<template.length; i++){
                res.push([
                    source.x + template[i][0]*p_dx + template[i][1]*s_dx,
                    source.y + template[i][0]*p_dy + template[i][1]*s_dy,
                ]);
            }
            return poly_d_fn(res);
        };
    }
    var dir_edge_template = [
        [0,4],[0.6,4],[0.6,12],[0.75,4],[1,4],
        [1,2],[0.75,2],[0.6,-6],[0.6,2],[0,2],
    ];
    
    function update_state(time){
        var cur_n_strengths = node_strengths[batch][time];
        var cur_n_ids = node_ids[batch][time];
        var cur_n_states = node_states[batch][time];
        var cur_e_strengths = edge_strengths[batch][time];
        var n_nodes = cur_n_strengths.length;
        var tmp_nodes = [];
        var tmp_edges = [];
        var tmp_display_edges = [];
        for(var i=0; i<n_nodes; i++){
            var n_strength = cur_n_strengths[i];
            tmp_nodes.push({
                state_index: i,
                strength: n_strength,
                color: colormap.node_id[batch][time][i],
                id:cur_n_ids[i],
                data: [n_strength].concat(cur_n_ids[i],cur_n_states[i].map(function(x){return (x+1.0)/2})),
                x: (i < data_nodes.length) ? data_nodes[i].x : options.width*Math.random(),
                y: (i < data_nodes.length) ? data_nodes[i].y : options.height*Math.random(),
            });
        }
        for(var i=0; i<n_nodes; i++){
            for(var j=0; j<n_nodes; j++){
                if(i==j)
                    continue;
                var eff_str = Math.min(1,cur_e_strengths[i][j].reduce(function(p,v){return p+v;},0));
                var eff_str_link = cur_e_strengths[i][j].map(function(v,etype){
                    return v * edge_strength_adjust[etype];
                }).reduce(function(p,v){return p+v;},0);
                eff_str = eff_str * cur_n_strengths[i] * cur_n_strengths[j];
                var c_edge = {
                    edge_index:i,
                    s:i,
                    d:j,
                    types: cur_e_strengths[i][j],
                    source:i,
                    target:j,
                    strength: eff_str,
                    link_force_strength: eff_str_link,
                    color: colormap.edge_type[batch][time][i][j],
                    data: cur_e_strengths[i][j],
                };

                if(eff_str>0.1){
                    tmp_edges.push(c_edge);
                }
                if(eff_str>0.03){
                    tmp_display_edges.push(c_edge);
                }
            }
        }
        for (var i=0; i<tmp_display_edges.length; i++){
            tmp_display_edges[i].source = tmp_nodes[tmp_display_edges[i].source];
            tmp_display_edges[i].target = tmp_nodes[tmp_display_edges[i].target];
        }
        data_nodes = tmp_nodes;
        data_edges = tmp_edges;
        display_edges = tmp_display_edges;

        force.nodes(data_nodes);
        force.force("link").links(data_edges)
            .strength(function(link,i){
                return params.linkStrength*link.link_force_strength;
            })
            .distance(params.linkDistance)
        force.force("charge").strength(function(node,i){
                return -params.charge*node.strength;
            });
        force.force("gravityX").strength(params.gravity);
        force.force("gravityY").strength(params.gravity);
        force.alpha(1).restart();

        for(var i=0; i<extra_snap_specs.length; i++){
            var spec = extra_snap_specs[i];
            var cur_force = extra_forces[i];
            cur_force.strength(function(node,ni){
                return spec.strength * node.id[spec.id];
            });
        }
        
        link_fwd=link_fwd.data(display_edges);
        link_fwd.exit().remove();
        link_fwd=link_fwd.enter().append("path")
            .classed("fwd",true)
            .merge(link_fwd)
            .attr('d',adjust(dir_edge_template,false))
            .attr('fill',colfn("color"))
            .attr('opacity',function(d,i){return d.strength});

        node=node.data(data_nodes);
        node.exit().remove();
        node=node.enter().append("circle")
            .merge(node)
            .attr('fill',colfn("color"))
            .attr('r','9')
            .attr('opacity',function(d,i){return d.strength});

        node.on("mouseover",do_focus)
        link_fwd.on("mouseover",do_focus)

        console.log("Updated!");
    }
    update_state(params.timestep);

    function redraw() {
      link_fwd=link_fwd.attr('d',adjust(dir_edge_template,false));

      node=node.attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    }
    force.on("tick", redraw);
    
    node.call(d3.drag()
          .container(svg.node())
          .subject(function(){return force.find(d3.event.x, d3.event.y)})
          .on("start", function dragstarted() {
              if (!d3.event.active) force.alphaTarget(0.6).restart();
              d3.event.subject.fx = d3.event.subject.x;
              d3.event.subject.fy = d3.event.subject.y;
          }).on("drag", function dragged() {
              d3.event.subject.fx = d3.event.x;
              d3.event.subject.fy = d3.event.y;
          }).on("end", function dragended() {
              if (!d3.event.active) force.alphaTarget(0);
              d3.event.subject.fx = null;
              d3.event.subject.fy = null;
          }));
    
    function update_focus(datalist){
        var div_w = (0.0 + width)/datalist.length;
        if(div_w>20) div_w = 20;
        focus_detail = focus_detail.data(datalist)
        focus_detail.exit().remove();
        focus_detail = focus_detail.enter().append("rect")
                    .merge(focus_detail)
                    .attr('fill',function(d){return d3.interpolateViridis(d).toString()})
                    .attr('width',div_w)
                    .attr('height',20)
                    .attr('x',function(d,i){return div_w*i})
                    .attr('y',height-20);
    }
    function do_focus(d){
        if(options.noninteractive)
            return;
        console.log("Focusing on ", d)
        update_focus(d.data);
    }
    
    var gui = new dat.GUI({ autoPlace: false });

    if(!options.noninteractive)
        el.insertBefore(gui.domElement, el.firstChild);
    
    gui.add(params,"linkDistance").min(0).max(200).onChange(function(value) {
        force.force("link").distance(value);
        force.alpha(1).restart();
    });
    gui.add(params,"linkStrength").min(0).max(1).onChange(function(value) {
        force.force("link").strength(function(link,i){
                return value*link.link_force_strength;
            })
        force.alpha(1).restart();
    });

    gui.add(params,"gravity").min(0).max(0.2).onChange(function(value) {
        force.force("gravityX").strength(value);
        force.force("gravityY").strength(value);
        force.alpha(1).restart();
    });
    gui.add(params,"charge").min(0).max(200).onChange(function(value) {
        force.alpha(1).restart();
    });

    if(options.jitter){
        gui.add(params,"jitterScale").min(0).max(200);
    }
    
    var last_timestep = params.timestep;
    gui.add(params,"timestep").min(0).max(max_time-1).step(1).onChange(function(value) {
        if(value != last_timestep){
            update_state(value);
            last_timestep = value;
        }
    });

    function noninteractive_update(){
        force.stop();
        var startTicks = options.fullAlphaTicks || 0;
        for(var i=0; i<startTicks; i++){
            force.alpha(1);
            force.tick();
        }
        while(force.alpha() > force.alphaMin()){
            force.tick();
        }
        redraw();
    }
    if(options.noninteractive){
        noninteractive_update();
        return function(){
            params.timestep++;
            if(params.timestep < max_time){
                update_state(params.timestep);
                noninteractive_update();
                return true;
            } else
                return false;
        }
    }
}
window._graph_display = _graph_display;
console.log("Loaded display_graph");
if(element)
    element.text("Done!");
});
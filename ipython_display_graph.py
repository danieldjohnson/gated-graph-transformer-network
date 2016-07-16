import numpy as np
from IPython.display import Javascript
import json

from IPython.core.display import HTML

from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils.validation import NotFittedError

id_projector = GaussianRandomProjection(n_components=3)
edge_projector = GaussianRandomProjection(n_components=3)

STATE_WIDTH = 50
def graph_display(states):
    clean_states = [x.tolist() for x in states]
    nstr, nid, nstate, estr = states
    flat_nid = nid.reshape([-1,nid.shape[-1]])
    flat_estr = estr.reshape([-1,estr.shape[-1]])
    flat_estr = flat_estr / (np.linalg.norm(flat_estr, axis=1, keepdims=True) + 1e-8)
    try:
        flat_transf_nid = id_projector.transform(flat_nid)
        flat_transf_estr = edge_projector.transform(flat_estr)
    except:
        flat_transf_nid = id_projector.fit_transform(flat_nid)
        flat_transf_estr = edge_projector.fit_transform(flat_estr)
    minlevel = np.min(flat_transf_nid)
    maxlevel = np.max(flat_transf_nid)
    node_colors = (flat_transf_nid-minlevel)/(maxlevel-minlevel)
    minlevel = np.min(flat_transf_estr)
    maxlevel = np.max(flat_transf_estr)
    edge_colors = (flat_transf_estr-minlevel)/(maxlevel-minlevel)
    colormap = {
        "node_id": node_colors.reshape(nid.shape[:-1] + (3,)).tolist(),
        "edge_type": edge_colors.reshape(estr.shape[:-1] + (3,)).tolist(),
    }
    return Javascript("window._graph_display({}, {}, element[0],0);".format(
            json.dumps(clean_states),
            json.dumps(colormap)))

JS_SETUP_STRING = """
require.config({
  paths: {
      d3: '//cdnjs.cloudflare.com/ajax/libs/d3/4.1.0/d3.min',
      dat: '//cdnjs.cloudflare.com/ajax/libs/dat-gui/0.5.1/dat.gui.min'
  }
});

require(['d3','dat'], function(d3,_ignored){
function _graph_display(states,colormap,el,batch){
    var node_strengths = states[0];
    var node_ids = states[1];
    var node_states = states[2];
    var edge_strengths = states[3];
    var max_time = node_strengths[batch].length;
    var svg = d3.select(el).append("svg").attr("width",500).attr("height",500);

    var node_map = {};
    var data_nodes = [];
    var data_edges = [];
    var display_edges = [];
    var selection_map, selection_options;
    var force = d3.forceSimulation()
                    .force("charge", d3.forceManyBody())
                    .force("link", d3.forceLink())
                    .force("gravityX", d3.forceX(250))
                    .force("gravityY", d3.forceY(250));
    
    var link_fwd = svg.append("g").selectAll("path.fwd");
    var node = svg.append("g").selectAll("circle");
    
    var focus_detail = svg.append("g").selectAll("rect");

    
    var params = {
        linkDistance: 80,
        gravity: 0.02,
        charge: 100,
        primarySelection: 'Active Selection',
        secondarySelection: 'Stored Selection 0',
        timestep:max_time-1,
    };
    
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
        [0,4],[0.6,4],[0.6,8],[0.75,4],[1,4],
        [1,2],[0.75,2],[0.6,-2],[0.6,2],[0,2],
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
                data: [n_strength].concat(cur_n_ids[i],cur_n_states[i].map(function(x){return (x+1.0)/2})),
                x: (i < data_nodes.length) ? data_nodes[i].x : 200+100*Math.random(),
                y: (i < data_nodes.length) ? data_nodes[i].y : 200+100*Math.random(),
            });
        }
        for(var i=0; i<n_nodes; i++){
            for(var j=0; j<n_nodes; j++){
                if(i==j)
                    continue;
                var eff_str = Math.min(1,cur_e_strengths[i][j].reduce(function(p,v){return p+v;},0));
                eff_str = eff_str * cur_n_strengths[i] * cur_n_strengths[j];
                var c_edge = {
                    edge_index:i,
                    s:i,
                    d:j,
                    source:i,
                    target:j,
                    strength: eff_str,
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
                return 0.1*link.strength;
            })
            .distance(params.linkDistance)
        force.force("charge").strength(function(node,i){
                return -params.charge*node.strength;
            });
        force.force("gravityX").strength(params.gravity);
        force.force("gravityY").strength(params.gravity);
        force.alpha(1).restart();
        
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

        console.log("Updated!");
    }
    update_state(params.timestep);

    force.on("tick", function() {
      link_fwd=link_fwd.attr('d',adjust(dir_edge_template,false));

      node=node.attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
    });
    
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
        var div_w = 500.0/datalist.length;
        if(div_w>20) div_w = 20;
        focus_detail = focus_detail.data(datalist)
        focus_detail.exit().remove();
        focus_detail = focus_detail.enter().append("rect")
                    .merge(focus_detail)
                    .attr('fill',function(d){return d3.interpolateViridis(d).toString()})
                    .attr('width',div_w)
                    .attr('height',20)
                    .attr('x',function(d,i){return div_w*i})
                    .attr('y',450);
    }
    update_focus([0,0.2,0.4,0.6,0.8,1.0]);
    function do_focus(d){
        console.log("Focusing on ", d)
        update_focus(d.data);
    }
    node.on("mouseover",do_focus)
    link_fwd.on("mouseover",do_focus)
    
    var gui = new dat.GUI({ autoPlace: false });
    el.insertBefore(gui.domElement, el.firstChild);
    
    gui.add(params,"linkDistance").min(0).max(200).onChange(function(value) {
        force.force("link").distance(value);
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
    
    var last_timestep = params.timestep;
    gui.add(params,"timestep").min(0).max(max_time-1).step(1).onChange(function(value) {
        if(value != last_timestep){
            update_state(value);
            last_timestep = value;
        }
    });
}
window._graph_display = _graph_display;
element.text("Done!");
});
"""

def setup_graph_display():
    return Javascript(JS_SETUP_STRING)
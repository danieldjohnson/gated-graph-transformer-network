#! phantomjs
function assert(condition, message) {
    if (!condition) {
        console.error(message);
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

var system = require('system');
var visdir = system.args[1];
var process = require("child_process");
var webPage = require('webpage');
var fs = require('fs');

console.log("Running python display script...");
process.execFile("python3", ["-m", "display.display_graph", visdir], null, function(err, stdout, stderr){
    console.log("Parsing...");
    var params_obj = JSON.parse(stdout);
    if(visdir.charAt(visdir.length-1) == fs.separator)
        visdir = visdir.substr(0, visdir.length-1);
    var imgdir = visdir + fs.separator + "generated_images"
    console.log("Creating images directory "+imgdir);
    assert(fs.isDirectory(imgdir) || fs.makeDirectory(imgdir), "Failed to make directory!");

    params_obj.options.noninteractive = true;
    params_obj.options.timestep = 0;

    console.log("Starting image generation...");
    var page = webPage.create();
    page.viewportSize = { width: (params_obj.options.width || 500), height: (params_obj.options.height || 500) };
    page.onConsoleMessage = function(msg) {
        // console.log("Page says: ", msg);
        if(msg == "Loaded display_graph"){
            page.evaluate(function(params_obj){
                window.next_fn = window._graph_display(params_obj.states, params_obj.colormap, document.body, 0, params_obj.options);
            }, params_obj);
            for(var i=0; true; i++){
                console.log("Writing image ", i);
                page.render(imgdir + fs.separator + i + '.png');
                var has_more = page.evaluate(function(){
                    return window.next_fn();
                });
                if(!has_more)
                    break;
            }
            phantom.exit();
        }
    }
    page.includeJs("https://cdnjs.cloudflare.com/ajax/libs/require.js/2.2.0/require.min.js", function(){
        page.injectJs("display/display_graph.js");
    })
})
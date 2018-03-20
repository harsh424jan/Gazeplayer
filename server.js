
var PythonShell = require('python-shell')
var express = require('express')
var path = process.cwd()
var app = express()

var bodyParser = require('body-parser');


app.use(express.static('views'))
var port = process.env.PORT || 8090;

app.use(bodyParser.urlencoded({extended:false}))

app.get('/',function(req,res){
res.sendFile(path.join(__dirname+'/view.html'));
})
	
app.post('/', function(req,res) {


var options = {
    mode: 'text',
    pythonPath: 'python2.7',
    args: []
};
res.sendFile(path+'/views/view.html');

PythonShell.run('gaze_player_final2.py', options, function (err, results) {
    if (err) throw err;
    // results is an array consisting of messages collected during execution
    console.log('results: %j',results);
    
});

});        
   


app.listen(8090);
console.log("Server Started.")


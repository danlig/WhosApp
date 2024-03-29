const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const app = express();
const ejs = require("ejs");
const axios = require("axios");
var jsdom = require("jsdom");
const { JSDOM } = jsdom;
const { window } = new JSDOM();
const { document } = (new JSDOM("")).window;
global.document = document;



var $ = jQuery = require("jquery")(window);

app.set("view engine", "ejs");
app.engine("ejs", require("ejs").__express);
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.urlencoded({ extended: false }));

const backend = "http://localhost:5000/WhosApp";

app.get("/", (request, response) => {
    response.render("index");
});

app.post("/newUserMessage", (req, res) => {
    let text = req.body.text;

    text = text.replace(/\\n/g, "<br>")
    res.render("user-bubble.ejs", {userText: text});

});

app.post("/getResponse", (req, res) => {
    let userText = req.body.text

    let data = {
        text: userText
    }

    try {
        console.log("Sending request to " + backend + "\nWith text: " + userText + "\n");

        axios.post(backend, data)
        .then(function (response) {

            result = response.data;
            console.log("Response from backend: " + result + "\n");

            ejs.renderFile('./views/bot-bubble.ejs', {single : result["single"], mappedUsers : result["mappedUsers"]}, (err, str) => {
                if (err) {
                    console.log(err);
                    res.status(500).send({ error: 'Errore nel caricamento del template' });
                } else {
                    let unrelatedData = {
                        mappedUsers: result["mappedUsers"],
                        average: result["average"]
                    };

                    res.send({ template: str, data: unrelatedData });
                }
              });

            // res.render("bot-bubble.ejs", {single : result["single"], average : result["average"], mappedUsers : result["mappedUsers"]});
        })
        .catch(function (error) {
            console.log("Error from backend: " + error + "\n")
            
            res.render("bot-bubble.ejs", {responseText : "<b>ERRORE</b>"});
        });
        
        

      } catch (error) {
        console.log(error);
        res.status(400).send("Errore nella comunicazione con il modello");
      }
})

app.listen(port = 3000, () => {
    console.log("Server avviato su porta " + port);
});
var buttonPOST = document.getElementById("buttonPOST");
var buttonGET = document.getElementById("getPOST");

var getLastUrl = function(callback) {
    chrome.history.search({
        text: '',
        maxResults: 8
    }, function(results) {
        // console.log(results);
        // console.log(results[0].url);
        callback(results);
    });
};

if (buttonPOST != null) {
    buttonPOST.addEventListener("click", function(e){
        e.preventDefault();
        getLastUrl(function(lastUrl) {
            // now you can use lastUrl here
            var test1 = lastUrl;
            // test 1 is a string of urls joined by " , "
            test1 = test1.map(e => e.url).join(" , ");
            // POST to local host -- here we are posting a string of urls
            fetch('http://3.18.102.43:8080/store', {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(test1),
                //headers: {"Content-type": "application/json; charset=UTF-8"}
            }).then(res => {
                console.log("Request complete! response:", res);
            });
        });
    });
}

if (buttonGET != null) {
    buttonGET.addEventListener("click", function(e) {
        e.preventDefault();
        fetch('http://3.18.102.43:8080/test')
            .then(response => response.json())
            .then(data => console.log(data));
    });
}

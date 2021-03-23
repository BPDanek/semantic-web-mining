let sendGraphButton = document.getElementById('sendGraphButton');

sendGraphButton.onclick = function(element) {

    chrome.runtime.sendMessage({sendToDB: true}, function(response) {
        console.log(response)
    });

    // put history data in sync
};
chrome.runtime.onInstalled.addListener(function() {
    chrome.history.search({
        'text': '',              // Return every history item....
        'maxResults': 100  // that was accessed less than one week ago.
    }, function(historyItem) {
        console.log(historyItem)
    });
});

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.sendToDB == true)
            console.log("sent response")
            sendResponse({response: ("sendToDB reads TRRRRRRUUUEE ")});
    }
);
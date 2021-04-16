// Initialize button with user's preferred color
let sendBUTTON = document.getElementById("sendPOSTB");

var getLastUrl = function(callback) {
  chrome.history.search({
      text: '',
      maxResults: 8
  }, function(results) {
      console.log(results);
      console.log(results[0].url);
      callback(results);
    });
};

function sendPOST1() {
  getLastUrl(function(lastUrl) {
    // now you can use lastUrl here
    var test1 = lastUrl;
    // test 1 is a string of urls joined by " , "
    test1 = test1.map(e => e.url).join(" , ");
    // POST to local host -- here we are posting a string of urls
    fetch('http://127.0.0.1:5000/', {
      method: "POST",
      body: JSON.stringify(test1),
      headers: {"Content-type": "application/json; charset=UTF-8"}
    })
  });
}


// When the button is clicked, inject setPageBackgroundColor into current page
sendBUTTON.addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: sendPOST1,
  });
});

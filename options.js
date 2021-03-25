// button to add to place within 'actionButtonContainer'
let button_element = document.createElement("button")
button_element.innerText = 'smores?'
button_element.addEventListener('click', () => {validator('smores.')})


// put the button_element inside this element (which exists in the options.html page)
document.getElementById("actionButtonContainer").appendChild(button_element)

// should output: {storage_phrase: "smores."}. If it does not may need to clear() (see below)
function validator(phrase_to_store) {
    // chrome.storage.sync.clear();

    // set multiple items: set(items: object, callback: function) => {...}
    chrome.storage.sync.set({storage_phrase: phrase_to_store}, function() {
        console.log("set phrase_to_store parameter in chrome.storage.sync")
    })

    chrome.storage.sync.get(null, function(result) {
        console.log("retrieved from storage:", result)
    })



}
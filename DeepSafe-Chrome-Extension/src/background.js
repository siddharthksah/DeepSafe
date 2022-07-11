chrome.browserAction.onClicked.addListener(function(activeTab){
    var newURL = "https://www.siddharthsah.com/deepsafe-webapp/";
    chrome.tabs.create({ url: newURL });
  });
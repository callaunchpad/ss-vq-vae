String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) { 
    return typeof args[number] != 'undefined'
        ? args[number]
        : match
    ;
    });
};

function adjustVolumes(index, val) {
    var target_file = Math.round(val / 10)
    audioMergers[index].setVolume(1, target_file);
    for(var i = 0; i <= 10; i++) {
        if (i === target_file) continue;
        audioMergers[index].setVolume(0, i);
    }
    /*
    if (val == 100) {
        audioMergers[index].muteAll()
        audioMergers[index].setVolume(1, 10)
    } else {
        firstFile = Math.floor(val / 10)
        volume = (val % 10) / 10
        secondFile = Math.floor(val / 10 + 1)
        
        for(var i = 0; i <= 10; i++) {
            if (i == firstFile || i == secondFile) continue
            audioMergers[index].setVolume(0, i);
        }
        audioMergers[index].setVolume(1 - volume, firstFile);
        audioMergers[index].setVolume(volume, secondFile);
    }*/
}

function stopAllAudios() {
    $('audio').each(function() {
        this.pause()
        this.currentTime = 0
    })
    $('.btn-file').each(function() {
        makeBtnGreen(this)
    })
}

function startPlayback(index) {
    stopAllAudios()
    audioMergers[index].setTime(0)
    adjustVolumes(index, 100-$('#slider-'+index).val())
    audioMergers[index].play()
}
function stopPlayback() {
    audioMergers.forEach(function(audioMerger) {
        audioMerger.pause()
        $('.btn-inter').each(function() {
            makeBtnGreen(this)
        })
    })
}

function togglePlayback(btn) {
    var index = parseInt(btn.getAttribute("index"))
    if (audioMergers[index].isPaused()) {
        stopPlayback()
        startPlayback(index)
        makeBtnRed(btn)
    } else {
        stopPlayback()
        makeBtnGreen(btn)
    }
}

function showPlaybackBtns(id) {
    $('#controls-'+id).show();
}

function playButtonClicked(btn) {
    var linkedAudioName = btn.getAttribute("linked-audio")
    var audio = document.getElementById(linkedAudioName);
    if (audio.paused) {
        stopAllAudios()
        stopPlayback()
        audio.play();
        makeBtnRed(btn)
    } else {
        audio.pause();
        audio.currentTime = 0
        makeBtnGreen(btn)
    }
}
function makeBtnRed(btn) {
    btn.classList.add('btn-danger')
    btn.classList.remove('btn-success')
    btn.children[0].classList.add('fa-stop')
    btn.children[0].classList.remove('fa-play')
}
function makeBtnGreen(btn) {
    btn.classList.remove('btn-danger')
    btn.classList.add('btn-success')
    btn.children[0].classList.add('fa-play')
    btn.children[0].classList.remove('fa-stop')
}
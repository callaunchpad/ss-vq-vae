class AudioMerger {
    constructor(files) {
        this.files = files;
        this.audios = files.map(file => new Audio(file));

        var AudioContext = window.AudioContext || window.webkitAudioContext;
        var ctx = new AudioContext();
        this.merger = ctx.createChannelMerger(this.audios.length);
        this.merger.connect(ctx.destination);

        this.gains = this.audios.map(audio => {
            var gain = ctx.createGain();
            var source = ctx.createMediaElementSource(audio);
            source.connect(gain);
            gain.connect(this.merger);
            return gain;
        });
        this.buffered = false;

        var load = files.length;
        this.audios.forEach(audio => {
            audio.addEventListener("canplaythrough", () => {
                load--;
                if (load === 0) {
                    this.buffered = true;
                    if (this.bufferCallback != null) this.bufferCallback();
                }
            });
        });
        this.audios[0].addEventListener("ended", () => {
            makeBtnGreen(document.getElementById('btn2'))
        })
    }

    onBuffered(callback) {
        if (this.buffered) callback();
        else this.bufferCallback = callback;
    }

    play() {
        this.audios.forEach(audio => audio.play());
    }

    pause() {
        this.audios.forEach(audio => audio.pause());
    }

    isPaused() {
        return this.audios[0].paused
    }

    getTime() {
        return this.audios[0].currentTime;
    }

    setTime(time) {
        this.audios.forEach(audio => audio.currentTime = time);
    }

    getDelay() {
        var times = [];
        for (var i = 0; i < this.audios.length; i++) {
        times.push(this.audios[i].currentTime);
        }

        var minTime = Math.min.apply(Math, times);
        var maxTime = Math.max.apply(Math, times);
        return maxTime - minTime;
    }

    setVolume(volume, audioID) {
        this.gains[audioID].gain.value = volume;
    }

    muteAll() {
        for (var i = 0; i < this.audios.length; i++)
            this.setVolume(0, i)
    }
}
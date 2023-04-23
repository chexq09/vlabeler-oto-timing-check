// delete temp file
let tempFile = pluginDirectory.resolve("temp.wav")
console.log('path of the temp file: ' + tempFile.getAbsolutePath())
if (tempFile.delete()) {
    console.log('temp file deleted')
}

// get exe path
let mergeExePath = params["mergeExePath"]
let mergeExeFile = File.fromPath(mergeExePath)

// check exe
let isMergeExeValid = true
if (!mergeExeFile.isFile()) {
    isMergeExeValid = false
} else {
    let mergeExeExtension = mergeExeFile.getExtension()
    if (Env.isWindows()) {
        if (mergeExeExtension !== "exe") {
            isMergeExeValid = false
        }
    } else {
        if (mergeExeExtension !== "") {
            isMergeExeValid = false
        }
    }
}

// error if exe is invalid
if (!isMergeExeValid) {
    error({
        en: `The given metronome synthesis engine is not a valid executable file: ${mergeExePath}`,
        zh: `给定的节拍器合成引擎不是有效的可执行文件: ${mergeExePath}`,
        ja: `指定されたメトロノーム合成エンジンは有効な実行ファイルではありません: ${mergeExePath}`
    })
}

// make exe executable if needed
if (!Env.isWindows()) {
    executeCommand("chmod", "+x", mergeExePath)
}

// check metronome file
let metronomePath = params["metronomePath"]
let mergeExeExtension = File.fromPath(metronomePath).getExtension()
if (mergeExeExtension != "wav") {
    error({
        en: `The given metronome is not a WAV file: ${metronomePath}`,
        zh: `给定的节拍器音频不是WAV文件: ${metronomePath}`,
        ja: `指定されたメトロノームはWAVファイルではありません: ${metronomePath}`
    })
}

let args = []
args.push(mergeExePath)

// metronome options
args.push(metronomePath)
args.push(params["metronomeCount"].toString())
args.push(params["bmp"].toString())
args.push(params["metronomeWeight"].toString())

// sample file
let entry = entries[currentEntryIndex]
let sampleDirectory = File.fromPath(module.sampleDirectory)
let sampleFile = sampleDirectory.resolve(entry.sample)
let sampleFilePath = sampleFile.getAbsolutePath()

args.push(sampleFilePath)

// location
let start = entry.points[3]
let offset = start - entry.points[1]
let end = entry.end
if (offset > 0) {
    error({
        en: `The offset is greater than 0: ${offset}`,
        zh: `偏移量大于0: ${offset}`,
        ja: `オフセットが0より大きいです: ${offset}`
    })
}

args.push(start.toString())
args.push(end.toString())
args.push(offset.toString())

// output
args.push(tempFile.getAbsolutePath())

let result = executeCommand(...args)

if (result === 0 && tempFile.exists()) {
    requestAudioFilePlayback(tempFile.getAbsolutePath())
}

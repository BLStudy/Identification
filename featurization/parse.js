const fs = require('fs');
const process = require('process');
const bsor = require('./open-replay-decoder.js');
const features = require('./features.js');
const utils = require('./utils.js');

function getSamples(user, set) {
  const samples = [];
  const replays = user[1][set];
  for (const replay of replays) {
    if (replay.includes('bsor')) {
      const file = fs.readFileSync("Z:/beatleader/replays/" + replay);
      try {
        const data = bsor.decode(file.buffer);
        if (data && data.frames && data.frames.length && data.notes && data.notes.length && data.info) {
          // utils.derive(data.frames);
          const endTime = data.frames[data.frames.length - 1].time;
          for (const note of data.notes) {
            if (note.noteCutInfo && note.noteCutInfo.speedOK && note.noteCutInfo.directionOK && note.noteCutInfo.saberTypeOK && note.eventTime > 1 && note.eventTime < (endTime - 1)) {
              let sample = [user[0]];
              sample = sample.concat(features.makeNoteFeatures(note));
              const framesBefore = utils.fastTimeSlice(data.frames, note.eventTime - 1, note.eventTime);
              sample = sample.concat(features.makeMotionFeatures(framesBefore));
              const framesAfter = utils.fastTimeSlice(data.frames, note.eventTime, note.eventTime + 1);
              sample = sample.concat(features.makeMotionFeatures(framesAfter));
              if (!sample.includes(null) && !sample.includes(undefined) && !sample.includes(NaN)) {
                samples.push(sample);
              }
            }
          }
        }
      } catch (err) { console.error(err) }
    }
  }
  return samples;
}

function handleUser(user, set, count) {
  if (fs.existsSync('./users/' + user[0] + '/' + set + '/0.csv')) return;
  if (!fs.existsSync('./users/' + user[0])) fs.mkdirSync('./users/' + user[0]);
  if (!fs.existsSync('./users/' + user[0] + '/' + set)) fs.mkdirSync('./users/' + user[0] + '/' + set);

  const samples = getSamples(user, set, count);
  utils.shuffle(samples);
  const selected = samples.slice(0, count);

  const chunkSize = 10;
  let j = 0;
  for (let i = 0; i < selected.length; i += chunkSize) {
      const chunk = selected.slice(i, i + chunkSize);
      fs.writeFileSync('./users/' + user[0] + '/' + set + '/' + j + '.csv', chunk.map(r => r.join(",")).join("\n") + "\n");
      j++;
  }
}

const id = parseInt(process.argv[2]);

const all = fs.readFileSync('users.txt', 'utf8').split("\n").filter(a=>a);
const split = JSON.parse(fs.readFileSync('split.json', 'utf8'));

for (let i = 0; i < all.length; i++) {
  if (i % 32 == id) {
    try {
      const user = all[i];
      const file = split[user].train[0];
      const samples = [];
      const raw = fs.readFileSync("Z:/beatleader/replays/" + file);
      const data = bsor.decode(raw.buffer);

      const endTime = data.frames[data.frames.length - 1].time;
      for (const note of data.notes) {
        if (samples.length > 10) break;
        if (note.noteCutInfo && note.noteCutInfo.speedOK && note.noteCutInfo.directionOK && note.noteCutInfo.saberTypeOK && note.eventTime > 1 && note.eventTime < (endTime - 1)) {
          let height = null;
          if (data.info.height) {
            height = data.info.height;
          } else if (data.heights.length > 0) {
            let heights = data.heights.filter(a => a.time < note.eventTime);
            height = heights[heights.length - 1].height;
          }
          let sample = [height];
          sample = sample.concat(features.makeNoteFeatures(note));
          const framesBefore = utils.fastTimeSlice(data.frames, note.eventTime - 1, note.eventTime);
          sample = sample.concat(features.makeMotionFeatures(framesBefore));
          const framesAfter = utils.fastTimeSlice(data.frames, note.eventTime, note.eventTime + 1);
          sample = sample.concat(features.makeMotionFeatures(framesAfter));
          if (!sample.includes(null) && !sample.includes(undefined) && !sample.includes(NaN)) {
            samples.push(sample);
          }
        }
      }
      fs.writeFileSync('./out/' + user + '.csv', samples.map(r => r.join(",")).join("\n") + "\n");
    } catch (err) { console.error(err) }
  }
}

for (let i = 0; i < all.length; i++) {
  if (i % 32 == id) {
    const user = all[i];
    handleUser(user, 'train', 1000);
    handleUser(user, 'test', 1000);
    handleUser(user, 'validate', 1000);
    console.log(i / all.length);
  }
}

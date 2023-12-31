from vad import VoiceActivityDetector
import argparse
import json

def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze input wave-file and save detected speech interval to json file.')
    parser.add_argument('--inputfile', metavar='./tilopa-by_the_way-05-by_the_ganges-175-204.wav', default='./tilopa-by_the_way-05-by_the_ganges-175-204.wav',
                        help='the full path to input wave file')
    parser.add_argument('--outputfile', metavar='./result1.json', default='./result1.json',
                        help='the full path to output json file to save detected speech intervals')
    print(parser)
    args = parser.parse_args()
    
    v = VoiceActivityDetector(args.inputfile)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    print(speech_labels)
    
    save_to_file(speech_labels, args.outputfile)

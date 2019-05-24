from flask import Flask, jsonify, abort, request, make_response, url_for
from datetime import datetime
from ab_one_class_svm import OneClassSVM_Ab

app = Flask(__name__, static_url_path = "")

HMAP = {}
model = None

def make_features(hmap, now):
    temps = []
    d_cnt = 0
    m_cnt = 0
    a_cnt = 0
    for k in list(hmap.keys()):
        if (now - k).seconds > 100:
            del hmap[k]
        else:
            temps.append(int(hmap[k]['temperature']))
            if hmap[k]['door_status'] == 'open':
                d_cnt += 1
            if hmap[k]['motion'] == 'active':
                m_cnt += 1
            if hmap[k]['acceleration'] == 'active':
                a_cnt += 1
    t_ch_f = len(set(temps))
    t_avg_f = sum(temps) / len(temps)

    return (t_ch_f, t_avg_f, d_cnt, m_cnt, a_cnt)


def make_timestamp(time):
    h = time.hour
    m = time.minute
    s = time.second
    return ((h * 60 + m) * 60 + s) // 1200


    
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

@app.route('/post', methods = ['POST'])
def get_status():
    global HMAP
    global model

    if not request.json:
        abort(400)
    motion = accel = door = temp = time = None
    if 'motion' in request.json:
        motion = request.json['motion']
    if 'acceleration' in request.json:
        accel = request.json['acceleration']
    if 'door_status' in request.json:
        door = request.json['door_status']
    if 'temperature' in request.json:
        temp = request.json['temperature']
    if 'time' in request.json:
        # time = datetime.strptime(request.json['time'], '%Y-%m-%d %H:%M:%S')
        time = datetime.strptime(request.json['time'][4:], '%b %d %H:%M:%S UTC %Y')

    if motion and accel and door and temp and time:
        HMAP[time] = {}
        HMAP[time]['motion'] = motion
        HMAP[time]['acceleration'] = accel
        HMAP[time]['door_status'] = door
        HMAP[time]['temperature'] = temp
        timestamp_f = make_timestamp(time)
        (t_ch_f, t_avg_f, d_cnt_f, m_cnt_f, a_cnt_f) = make_features(HMAP, time)
        output = model.predict(t_ch_f, t_avg_f, d_cnt_f, m_cnt_f, a_cnt_f, timestamp_f)
    else:
        output = "invalid input"
    return jsonify({ 'data': str(output) }), 201

    
if __name__ == '__main__':
    model = OneClassSVM_Ab('model.p')
    app.run(debug = True)

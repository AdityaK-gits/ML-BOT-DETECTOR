import http from 'k6/http';
import { sleep, check } from 'k6';
import { Trend } from 'k6/metrics';

export let options = {
  stages: [
    { duration: '10s', target: 20 },
    { duration: '30s', target: 200 }, // ramp
    { duration: '60s', target: 200 }, // sustain
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<400'],
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || '';
const headers = API_KEY ? { 'X-API-Key': API_KEY, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };

const body = JSON.stringify({
  user_id: 'k6_user',
  timestamp: new Date().toISOString(),
  request_path: '/home',
  request_duration: 0.25,
  mouse_movements: [],
  click_pattern: [],
  typing_speed: 0,
  scroll_behavior: { speed: 5, direction: 'down' }
});

export default function () {
  const res = http.post(`${BASE_URL}/detect-bot`, body, { headers });
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(1);
}

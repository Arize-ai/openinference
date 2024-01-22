#!/bin/bash

# This run script is used to run the backend and frontend apps in the background by docker

# Run the backend app from the /backend in the background
cd backend && python main.py &

# Run the frontend app from the /frontend in the background
cd frontend && npm run dev &

pid=
trap 'echo SIGINT; [[ $pid ]] && kill $pid; exit' SIGINT
trap 'echo SIGTERM; [[ $pid ]] && kill $pid; exit' SIGTERM
echo Starting script
sleep 10000 & pid=$!
wait
pid=
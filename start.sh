#!/bin/bash
uvicorn x:app --host 0.0.0.0 --port $PORT

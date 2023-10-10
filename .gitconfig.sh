#!/bin/sh
# @version : 0.1

# Initialize github hooks
git config core.hooksPath .githooks
chmod -R +x .githooks

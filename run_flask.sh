export PYTHONPATH="${PYTHONPATH}:`pwd`"
echo "${PYTHONPATH}:`pwd`"
export FLASK_ENV=development
flask run -h localhost -p 41373
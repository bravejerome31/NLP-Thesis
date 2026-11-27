from app import create_app

app = create_app()

if __name__ == '__main__':
    # use_reloader=False prevents the Windows PyTorch multiprocessing crash
    app.run(debug=True, use_reloader=False, port=5000)
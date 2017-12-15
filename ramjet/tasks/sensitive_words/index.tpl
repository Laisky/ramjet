<html>
    <head>
        <title>Image Detect Description</title>
    </head>
    <body>
        <pre>
            Run in command lines:
            ::
                $ pip install httpie
                $ http post app.laisky.com/image/detect/ urls:='["https://s3-us-west-1.amazonaws.com/movoto-data/demo_100x150.jpeg"]'

            Or:
            ::
                import requests

                url = 'app.laisky.com/image/detect/'
                data = {
                    'urls': [
                        # maximum to 5 picture urls
                        xxx,
                        xxx,
                    ]
                }
                resp = requests.post(url, json=data)
                print(resp.json())
        </pre>
    </body>
</html>

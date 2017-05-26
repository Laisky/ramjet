'use strict';

const path = require('path'),
    webpack = require('webpack'),
    ModernizrPlugin = require('modernizr-webpack-plugin'),
    HtmlWebpackPlugin = require('html-webpack-plugin'),
    ExtractTextPlugin = require('extract-text-webpack-plugin');


const modernizrConfig = {
    "filename": "modernizr.js",
    'options': [
        'setClasses',
        'html5printshiv'
    ],
    'feature-detects': [
        "inputtypes",
        "network/connection",
        "touchevents"
    ],
    "minify": {
        "output": {
            "comments": false,
            "beautify": false,
            "screw_ie8": true
        }
    }
}
const node_modules = path.resolve(__dirname, 'node_modules');

var stripLogger = 'strip-loader?strip[]=console.error' +
    '&strip[]=console.log' +
    '&strip[]=console.warn'


module.exports = {
    // The base directory for resolving the entry option
    context: __dirname,

    entry: {
        app: './ramjet/tasks/static/src/jsx/router.jsx',
        vendor: ['react', 'react-dom', 'react-redux']
    },

    // Various output options, to give us a single bundle.js file with everything resolved and concatenated
    output: {
        path: path.resolve(__dirname, './ramjet/tasks/static/dist'),
        filename: '[name]-[hash].js',
        publicPath: '/static/',
        pathinfo: true
    },

    resolve: {
        // Directories that contain our modules
        modules: [path.resolve(__dirname, "lib"), "node_modules"],
        descriptionFiles: ["package.json"],
        moduleExtensions: ["-loader"],
        // Extensions used to resolve modules
        extensions: ['.js', '.jsx', '.react.js', '.scss', '.css']
    },

    module: {
        noParse: [
            path.resolve(node_modules, 'react/dist/react.min.js'),
            path.resolve(node_modules, 'react-dom/dist/react-dom.min.js'),
            path.resolve(node_modules, 'react-router/umd/react-router.min.js'),
            path.resolve(node_modules, 'react-redux/dist/react-redux.min.js')
        ],
        rules: [
            {
                test: /\.scss$/,
                use: [{
                    loader: "style-loader" // creates style nodes from JS strings
                }, {
                    loader: "css-loader" // translates CSS into CommonJS
                }, {
                    loader: "sass-loader" // compiles Sass to CSS
                }]
            },
            {
                test: /\.jsx$/,
                use: ['babel-loader', stripLogger, stripLogger],
                exclude: [/node_modules/]
            },
        ],
    },

    plugins: [
        // handles creating an index.html file and injecting assets. necessary because assets
        // change name because the hash part changes. We want hash name changes to bust cache
        // on client browsers.
        new HtmlWebpackPlugin({
            template: './ramjet/tasks/static/index.tpl.html',
            inject: 'body',
            filename: 'index.html'
        }),
        new webpack.DefinePlugin({
            'process.env': {
                NODE_ENV: '"production"'
            }
        }),
        new ModernizrPlugin(modernizrConfig),
        new webpack.optimize.CommonsChunkPlugin({
            name: "vendor",
            filename: "vendor.js"
        }),
        new webpack.optimize.UglifyJsPlugin({
            compressor: {
                pure_getters: true,
                unsafe: true,
                unsafe_comps: true,
                warnings: false
            },
            output: {
                comments: false
            }
        }),
        new webpack.LoaderOptionsPlugin({
            minimize: true,
            debug: false
        }),
        new ExtractTextPlugin('[name]-[hash].css')
    ],

    // Include mocks for when node.js specific modules may be required
    node: {
        fs: 'empty',
        vm: 'empty',
        net: 'empty',
        tls: 'empty'
    }
};

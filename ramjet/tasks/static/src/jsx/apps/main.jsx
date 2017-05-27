'use strict';


import React from 'react';
import { Link } from 'react-router';

import { BaseComponent } from '../components/base.jsx';
import '../../sass/pages/webapps.scss';


export class App extends BaseComponent {
    static get contextTypes() {
        return {
            router: function () {
                return React.PropTypes.func.isRequired;
            }
        }
    };


    getScrollToTopHandle() {
        return (evt) => {
            if(evt.target.tagName.toUpperCase() != 'DIV' || evt.target.className.startsWith('gsc-')) return;
            $(document.body).animate({scrollTop: 0}, 500);

            return false;
        }
    };


    render() {
        let childContent;
        if(this.props.children) {
            childContent = this.props.children;
        }else {
            childContent = <article style={{'margin-top': '400px', 'text-aligin': 'center'}}>
              <p>Welcome Ramjet WebApps</p>
            </article>
        }

        console.log(childContent);

        return <div id="webapps" className="container-fluid">
          <div className="container-fluid">
            {/* page nav */}
            <nav className="navbar navbar-default navbar-fixed-top" onClick={this.getScrollToTopHandle()}>
              <div className="container-fluid">
                <div className="navbar-header">
                  <button type="button" className="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false">
                    <span className="sr-only">Toggle navigation</span>
                    <span className="icon-bar"></span>
                    <span className="icon-bar"></span>
                    <span className="icon-bar"></span>
                  </button>
                  <Link to={{ pathname: '/archives/1/' }} className="navbar-brand">WebApps</Link>
                </div>

                {/*<!-- Collect the nav links, forms, and other content for toggling -->*/}
                <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                  <ul className="nav navbar-nav">
                    <li className="dropdown">
                      <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Apps <span className="caret"></span></a>
                      <ul className="dropdown-menu">
                        <li><Link to="/webapps/">Home</Link></li>
                        <li role="separator" className="divider"></li>
                        <li><Link to="htts://blog.laisky.com/">Blog</Link></li>
                        <li><Link to="/webdemo/">Hello, World</Link></li>
                        <li><Link to="/twitter/login/">Login</Link></li>
                      </ul>
                    </li>
                    <li className="dropdown">
                      <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Documents <span className="caret"></span></a>
                      <ul className="dropdown-menu">
                        <li><Link to="//app.laisky.com/style-guide//">Style Guide</Link></li>
                      </ul>
                    </li>
                  </ul>
                </div>
              </div>
            </nav>

            {/* page modal */}
            <div className="modal fade" id="img-modal">
              <div className="modal-dialog" style={{'z-index': 1050, width: '800px'}}>
                <div className="modal-content">
                  <div className="modal-body" style={{padding: '0px'}}>
                    <img src="" alt="image" className="img-rounded" style={{'max-height': '800px', 'max-width': '800px'}} />
                  </div>
                </div>
              </div>
            </div>

            {/* page content */}
            <div ref="container" className='container' id="container">
              {childContent}
            </div>
          </div>
        </div>
    }
}

import React from "react";
import "./CameraFeed.css";
import { MQTT, HTTP, SETTINGS } from "../../constants/constants";
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert'
import mq from "../../MqttClient";

class CameraFeed extends React.Component {
  constructor( props ) {
    super( props );
    this.handleMqtt = this.handleMqtt.bind( this );
    this.mjpgSrc = HTTP.CAMERA_FEED;
    this.refreshImage = this.refreshImage.bind( this );
    this.mjpgSrc = HTTP.CAMERA_FEED;
    this.handleChangeStopSending = this.handleChangeStopSending.bind(this);
    this.handleChangeshowInferenceStats = this.handleChangeshowInferenceStats.bind(this);
    this.handleChangeMaxPeople = this.handleChangeMaxPeople.bind(this);
    this.lastvalue = 0;
    this.state = {
      mjpgSrc: this.mjpgSrc,
      stopSending: false,
      showInferenceStats: true,
      maxPeople: 5,
      show: false ,
      msg: "Testing alert",
      variant: "primary",
      fontColor: "--intel-charcoal"
    };
  }
  componentDidMount() {
    // register handler with mqtt client
    mq.addHandler( "person", this.handleMqtt );
  }

  componentWillUnmount() {
    mq.removeHandler( "person" );
  }
  handleMqtt( topic, payload ) {
    switch ( topic ) {
      case MQTT.TOPICS.PERSON:
        if(payload.count != this.lastvalue){
          if(this.state.maxPeople == payload.count){
            this.setState({ msg: "The room is fully crowded now, please close the door", show: true, variant: "warning", fontColor: "--intel-orange"});
          }
          else if(this.state.maxPeople < payload.count){
            this.setState({ msg: "The room has overpassed the maximum capacity, please evacuate people", show: true, variant: "warning", fontColor: "--intel-blood-orange"});
          }
          else if(payload.count >= parseInt(0.8*this.state.maxPeople) && payload.count < this.state.maxPeople){
            this.setState({ msg: "The room is getting the max capacity, please take this in consideration to avoid to crow the place", show: true, variant: "info", fontColor: "--intel-azure"});
          }
          else{
            this.setState({ msg: "", show: false, variant: "primary"});
          }
        }
        this.lastvalue = payload.count;
        break;
      default:
        break;
    }
  }

  handleChangeStopSending(evt) {
    this.setState({ stopSending: evt.target.checked });
    mq.publish("stopSending", {value:evt.target.checked});
  }
  handleChangeshowInferenceStats(evt) {
    this.setState({ showInferenceStats: evt.target.checked });
    mq.publish("showInferenceStats", {value:evt.target.checked});
  }
  handleChangeMaxPeople(evt) {
    this.setState({ maxPeople: evt.target.value });
  }
  refreshImage() {
    const d = new Date();
    this.setState( { mjpgSrc: `${ this.mjpgSrc }?ver=${ d.getTime() }` } );
  }

  render() {
    const width = SETTINGS.CAMERA_FEED_WIDTH;//640
    const imgStyle = { "maxWidth": `${ width }px` };
    return (
      <div className="camera-feed" >
        <div className="camera-feed-container">
          <font color="--intl-charcoal">
          <Form className="form-check-inline">
            <Form.Row>
              <Col>
                <Form.Label>Max number of people: </Form.Label>
              </Col>
              <Col>
                <Form.Control type="number" value={this.state.maxPeople} onChange={this.handleChangeMaxPeople} name="maxPeople"/>
              </Col>
            </Form.Row>
            <Form.Row>
              <Col>
                <Form.Label>Show video stats</Form.Label>
              </Col>
              <Col>
                <Form.Check type="checkbox" checked={this.state.showInferenceStats} onChange={this.handleChangeshowInferenceStats} name="showInferenceStats"/>
              </Col>
            </Form.Row>
            <Form.Row>
              <Col>
                <Form.Label>Stop Sending video? </Form.Label>
              </Col>
              <Col>
                <Form.Check type="checkbox" checked={this.state.stopSending} onChange={this.handleChangeStopSending} name="stopSending"/>
              </Col>
            </Form.Row>
          </Form>
          </font>
           <font color={this.state.fontColor}>
          <Alert show={this.state.show} className="alert-box" variant={this.state.variant}>{this.state.msg}</Alert>4
          </font>
          <img src={ this.state.mjpgSrc } alt="camera feed" style={ imgStyle } onClick={ this.refreshImage } className="camera-feed-img" />
        </div>
      </div>
    );
  }
}

export default CameraFeed;
